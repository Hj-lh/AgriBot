"""
AgriBot – FastAPI Main Application
====================================
Central server that exposes REST + streaming endpoints
for motor control, water pump, camera feed, and AI detection.

Run with:  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from components.motor import MotorController
from components.camera import RobotCamera
from components.waterpump import WaterPumpController
from components.ai import PlantDetector

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
    logger.info("All components ready ✔")

    yield  # ← app is running

    logger.info("Shutting down AgriBot components …")
    system["motor"].close()
    system["camera"].close()
    system["pump"].close()
    system["ai"].close()
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

def _mjpeg_generator():
    """Yield JPEG frames as an MJPEG stream."""
    camera: RobotCamera = system["camera"]
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.get("/camera/feed")
def video_feed():
    """Live MJPEG video stream."""
    return StreamingResponse(
        _mjpeg_generator(),
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
