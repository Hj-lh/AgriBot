"""
AgriBot – FastAPI Main Application
====================================
Central server that exposes REST + streaming endpoints
for motor control, water pump, camera feed, and AI detection.

Run with:  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

import cv2

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from components.motor import MotorController
from components.camera import RobotCamera
from components.camera_control import CameraPTZController
from components.waterpump import WaterPumpController
from components.ai import PlantDetector
from components.reid import PlantReID
from components.automatic import AutoNavigator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# System state — populated on startup, cleaned up on shutdown
# ------------------------------------------------------------------
system: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise hardware on startup, release on shutdown."""
    logger.info("Initialising AgriBot components …")
    system["motor"] = MotorController()
    system["camera"] = RobotCamera()
    system["pump"] = WaterPumpController()
    system["ai"] = PlantDetector()
    try:
        system["ptz"] = CameraPTZController()
    except Exception as e:  # noqa: BLE001
        logger.warning("PTZ camera control unavailable: %s", e)
        system["ptz"] = None
    system["reid"] = PlantReID()
    system["navigator"] = AutoNavigator(
        system["motor"],
        system["pump"],
        ptz=system["ptz"],
        reid=system["reid"],
    )
    logger.info("All components ready ✔")

    yield  # ← app is running

    logger.info("Shutting down AgriBot components …")
    system["navigator"].stop()
    system["motor"].close()
    system["camera"].close()
    system["pump"].close()
    system["ai"].close()
    if system.get("ptz") is not None:
        system["ptz"].close()
    logger.info("Shutdown complete")


app = FastAPI(title="AgriBot API", version="1.0.0", lifespan=lifespan)

# Allow the frontend to connect from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================================
# Root
# ==================================================================

@app.get("/")
def index():
    return {
        "message": "AgriBot System Online",
        "endpoints": {
            "camera": ["/camera/feed", "/camera/snapshot"],
            "motor": [
                "/motor/move?direction=forward&speed=0.8",
                "/motor/stop",
            ],
            "pump": ["/pump/on", "/pump/off", "/pump/status"],
            "ai": ["/ai/detect"],
            "ptz": [
                "/camera/ptz/pan?angle=45",
                "/camera/ptz/tilt?angle=-10",
                "/camera/ptz/zoom?level=2000",
                "/camera/ptz/move?pan_speed=50&tilt_speed=0",
                "/camera/ptz/look?direction=right&degrees=90",
                "/camera/ptz/center",
                "/camera/ptz/home",
                "/camera/ptz/stop",
                "/camera/ptz/position",
            ],
        },
    }


# ==================================================================
# Motor
# ==================================================================

@app.post("/motor/move")
def move_robot(
    direction: str = Query(..., description="forward | backward | left | right"),
    speed: float = Query(0.8, ge=0.0, le=1.0, description="Speed 0.0 – 1.0"),
):
    """Drive the robot in the given direction at the given speed."""
    motor: MotorController = system["motor"]

    actions = {
        "forward": motor.forward,
        "backward": motor.backward,
        "left": motor.left,
        "right": motor.right,
    }

    action = actions.get(direction)
    if action is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid direction: {direction}"},
        )

    action(speed)
    return {"status": "moving", "direction": direction, "speed": speed}


@app.post("/motor/stop")
def stop_robot():
    """Stop all motors immediately."""
    system["motor"].stop()
    return {"status": "stopped"}


# ==================================================================
# Camera
# ==================================================================

def _mjpeg_generator(mode: str = "manual"):
    """Yield JPEG frames as an MJPEG stream.
    In 'automatic' mode, YOLO detection boxes are drawn on each frame.
    """
    camera: RobotCamera = system["camera"]
    detector: PlantDetector = system["ai"]
    use_ai = mode == "automatic" and detector.enabled

    while True:
        if use_ai:
            # Get raw frame for AI processing
            raw_frame = camera.get_raw_frame()
            if raw_frame is None:
                continue

            detections = detector.detect(raw_frame)

            navigator = system["navigator"]
            if navigator.is_active:
                navigator.update_detections(
                    detections,
                    raw_frame.shape[1],
                    raw_frame.shape[0],
                    frame=raw_frame,
                )

            annotated = detector.annotate_frame(
                raw_frame,
                detections,
                watered_ids=navigator.watered_ids,
                watering=navigator.is_watering,
            )

            _, jpeg = cv2.imencode(".jpg", annotated)
            frame = jpeg.tobytes()
        else:
            frame = camera.get_frame()
            if frame is None:
                continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.get("/camera/feed")
def video_feed(
    mode: str = Query("manual", description="manual | automatic"),
):
    """Live MJPEG video stream. Use mode=automatic for AI detection overlay."""
    if mode == "automatic":
        system["navigator"].start()
    else:
        system["navigator"].stop()

    return StreamingResponse(
        _mjpeg_generator(mode),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/camera/snapshot")
def snapshot():
    """Return a single JPEG frame."""
    frame = system["camera"].get_frame()
    if frame is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "No frame available"},
        )
    return StreamingResponse(iter([frame]), media_type="image/jpeg")


# ==================================================================
# Water Pump
# ==================================================================

@app.post("/pump/on")
def pump_on():
    """Turn the water pump ON."""
    system["pump"].on()
    return {"status": "pump_on"}


@app.post("/pump/off")
def pump_off():
    """Turn the water pump OFF."""
    system["pump"].off()
    return {"status": "pump_off"}


@app.get("/pump/status")
def pump_status():
    """Check whether the pump is currently running."""
    return {"is_on": system["pump"].is_on}


# ==================================================================
# AI Detection
# ==================================================================

@app.get("/ai/detect")
def ai_detect():
    """
    Grab the latest camera frame, run YOLO detection, and
    return the list of detected objects.
    """
    detector: PlantDetector = system["ai"]

    if not detector.enabled:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "AI model not loaded"},
        )

    frame = system["camera"].get_raw_frame()
    if frame is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "No camera frame available"},
        )

    detections = detector.detect(frame)
    return {"status": "ok", "count": len(detections), "detections": detections}


# ==================================================================
# Camera PTZ (pan / tilt / zoom)
# ==================================================================

def _ptz_or_503():
    ptz: CameraPTZController | None = system.get("ptz")
    if ptz is None or not ptz.enabled:
        return None, JSONResponse(
            status_code=503,
            content={"status": "error", "message": "PTZ camera control not available"},
        )
    return ptz, None


@app.post("/camera/ptz/pan")
def ptz_pan(
    angle: float = Query(..., ge=-180.0, le=180.0, description="Absolute pan in degrees"),
):
    """Pan the camera to an absolute angle."""
    ptz, err = _ptz_or_503()
    if err is not None:
        return err
    ok = ptz.pan(angle)
    return {"status": "ok" if ok else "error", "pan": angle}


@app.post("/camera/ptz/tilt")
def ptz_tilt(
    angle: float = Query(..., ge=-90.0, le=90.0, description="Absolute tilt in degrees"),
):
    """Tilt the camera to an absolute angle."""
    ptz, err = _ptz_or_503()
    if err is not None:
        return err
    ok = ptz.tilt(angle)
    return {"status": "ok" if ok else "error", "tilt": angle}


@app.post("/camera/ptz/zoom")
def ptz_zoom(
    level: int = Query(..., ge=1, le=9999, description="Absolute zoom 1..9999"),
):
    """Set absolute zoom level."""
    ptz, err = _ptz_or_503()
    if err is not None:
        return err
    ok = ptz.zoom(level)
    return {"status": "ok" if ok else "error", "zoom": level}


@app.post("/camera/ptz/move")
def ptz_move(
    pan_speed: int = Query(0, ge=-100, le=100, description="Continuous pan speed"),
    tilt_speed: int = Query(0, ge=-100, le=100, description="Continuous tilt speed"),
):
    """Continuous PTZ motion until /camera/ptz/stop is called."""
    ptz, err = _ptz_or_503()
    if err is not None:
        return err
    ok = ptz.move_continuous(pan_speed, tilt_speed)
    return {"status": "ok" if ok else "error", "pan_speed": pan_speed, "tilt_speed": tilt_speed}


@app.post("/camera/ptz/stop")
def ptz_stop():
    """Stop any continuous PTZ motion."""
    ptz, err = _ptz_or_503()
    if err is not None:
        return err
    ok = ptz.stop_movement()
    return {"status": "ok" if ok else "error"}


@app.post("/camera/ptz/look")
def ptz_look(
    direction: str = Query(..., description="left | right | up | down | center"),
    degrees: float = Query(90.0, ge=0.0, le=180.0),
):
    """Convenience helper: look left/right/up/down by a number of degrees."""
    ptz, err = _ptz_or_503()
    if err is not None:
        return err

    actions = {
        "left":   lambda: ptz.look_left(degrees),
        "right":  lambda: ptz.look_right(degrees),
        "up":     lambda: ptz.look_up(degrees),
        "down":   lambda: ptz.look_down(degrees),
        "center": ptz.look_center,
    }
    action = actions.get(direction)
    if action is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid direction: {direction}"},
        )
    ok = action()
    return {"status": "ok" if ok else "error", "direction": direction, "degrees": degrees}


@app.post("/camera/ptz/center")
def ptz_center():
    """Re-center pan to 0°."""
    ptz, err = _ptz_or_503()
    if err is not None:
        return err
    ok = ptz.look_center()
    return {"status": "ok" if ok else "error"}


@app.post("/camera/ptz/home")
def ptz_home():
    """Recall the camera's configured home preset."""
    ptz, err = _ptz_or_503()
    if err is not None:
        return err
    ok = ptz.home()
    return {"status": "ok" if ok else "error"}


@app.post("/camera/ptz/preset")
def ptz_preset(
    name: str | None = Query(None, description="Preset name configured in the camera"),
    number: int | None = Query(None, ge=1, description="Preset index (1-based)"),
):
    """Recall a server preset position by name or by number."""
    ptz, err = _ptz_or_503()
    if err is not None:
        return err
    if name is None and number is None:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Provide either 'name' or 'number'"},
        )
    ok = ptz.goto_preset(name) if name is not None else ptz.goto_preset_number(number)
    return {"status": "ok" if ok else "error", "name": name, "number": number}


@app.get("/camera/ptz/position")
def ptz_position():
    """Return the current pan/tilt/zoom values reported by the camera."""
    ptz, err = _ptz_or_503()
    if err is not None:
        return err
    data = ptz.get_position()
    if data is None:
        return JSONResponse(
            status_code=502,
            content={"status": "error", "message": "Could not query PTZ position"},
        )
    return {"status": "ok", "position": data}
