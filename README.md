# 🌱 AgriBot Backend

A **FastAPI**-powered backend for a Raspberry Pi agricultural robot. Controls
motors, a water pump, a live AXIS PTZ camera feed, AI-based plant detection
with **ByteTrack** tracking, and a fully autonomous "find-and-water" loop —
all exposed as a REST API.

---

## 📁 Project Structure

```
Backend/
├── components/
│   ├── camera.py          # MJPEG camera capture (AXIS 213)
│   ├── camera_control.py  # PTZ HTTP control (AXIS VAPIX)
│   ├── ai.py              # YOLO + ByteTrack plant detection & annotation
│   ├── reid.py            # MobileNet-V3 appearance re-ID for watered plants
│   ├── motor.py           # Differential-drive motor control (PWM, 4× enable)
│   ├── waterpump.py       # Relay-driven water pump
│   └── automatic.py       # Autonomous navigation state machine
├── ai_models/
│   ├── best.pt            # ← Custom-trained model
│   └── yolo11n.pt         # ← Generic fallback
├── main.py                # FastAPI application (entry point)
├── requirements.txt       # Python dependencies
├── start.sh               # One-command startup script
└── README.md
```

---

## ⚙️ Hardware Setup

| Component             | GPIO Pin(s)             | Notes                                       |
|-----------------------|-------------------------|---------------------------------------------|
| Left motor (PWM)      | GPIO 12 / 13            | Forward / backward, hardware PWM 1 kHz      |
| Left motor enables    | GPIO 5 / 17             | BTS7960 R_EN / L_EN                          |
| Right motor (PWM)     | GPIO 22 / 23            | Forward / backward                           |
| Right motor enables   | GPIO 6 / 27             | BTS7960 R_EN / L_EN                          |
| Water pump relay      | GPIO 25                 | Active-low relay                             |
| AXIS 213 PTZ camera   | Network (HTTP)          | MJPEG + VAPIX at `169.254.138.53`            |

---

## 📋 System Prerequisites

Before installing Python packages, install these system-level dependencies on
the Raspberry Pi:

```bash
sudo apt update
sudo apt install -y swig liblgpio-dev python3-dev python3-venv
```

| Package          | Why it's needed                                    |
|------------------|----------------------------------------------------|
| `swig`           | Required to build the `lgpio` Python wheel          |
| `liblgpio-dev`   | C library for GPIO access (lgpio pin factory)       |
| `python3-dev`    | Python headers for compiling native extensions      |
| `python3-venv`   | Allows creating virtual environments                |

> **Note:** These only need to be installed once on a fresh Raspberry Pi OS.

---

## 🔧 Configuration

Camera host, credentials, and YOLO target class are hardcoded at the top of
their respective modules — edit them directly if your setup differs:

| Setting          | File                              | Default                  |
|------------------|-----------------------------------|--------------------------|
| PTZ host         | `components/camera_control.py`    | `169.254.138.53`         |
| PTZ user         | `components/camera_control.py`    | `root`                   |
| PTZ password     | `components/camera_control.py`    | `root`                   |
| PTZ HTTP timeout | `components/camera_control.py`    | `3.0` s                  |
| MJPEG source URL | `components/camera.py`            | `http://169.254.138.53/axis-cgi/mjpg/video.cgi` |
| YOLO target class| `components/ai.py`                | `plant`                  |
| ReID similarity  | `components/reid.py`              | `0.82` (cosine)          |

If the PTZ camera is unreachable on startup, the controller logs a warning and
disables itself; the rest of the bot keeps working and the autonomous navigator
falls back to motor-only scanning.

---

## 🚀 Quick Start

### 1. Transfer to Raspberry Pi

```bash
scp -r Backend/ pi@<pi-ip>:~/Desktop/
```

### 2. Install system dependencies

```bash
sudo apt update
sudo apt install -y swig liblgpio-dev python3-dev python3-venv
```

### 3. Place your YOLO model

```bash
cp best.pt ~/Desktop/Backend/ai_models/
```

### 4. Run

```bash
cd ~/Desktop/Backend
chmod +x start.sh
./start.sh
```

The script will:

- Create & activate the virtual environment (if first run)
- Install dependencies from `requirements.txt`
- Start the FastAPI server on `http://0.0.0.0:8000`
- Optionally start a **Cloudflare tunnel** for remote access

### Manual Start (without script)

```bash
cd ~/Desktop/Backend
source venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🤖 Autonomous Mode

When `/camera/feed?mode=automatic` is requested, the **AutoNavigator** state
machine takes over. Behaviour:

1. **Tracking** — picks the highest-confidence unwatered plant and turns/
   approaches based on its position (LEFT / CENTER / RIGHT zone).
2. **Watering** — once the bounding box fills ≥ 50 % of the frame and the
   plant is centred, the pump runs for 5 s. While watering, a
   `WATERING PLANT...` banner is overlaid on the video stream.
3. **Track-ID memory** — ByteTrack track IDs of watered plants are remembered
   in `watered_ids`; the annotator renders them in blue with a `WATERED`
   label, while fresh plants render in green with `NOT WATERED`.
4. **Appearance re-ID** — a MobileNet-V3 backbone crops each watered plant
   and stores a 1024-D unit-vector embedding. Subsequent detections whose
   appearance matches (cosine similarity ≥ 0.82) are *also* treated as
   watered, even after their ByteTrack ID is lost to occlusion, a 180°
   robot rotation, or a PTZ sweep. ReID disables itself gracefully if
   torch / torchvision aren't available; the bot falls back to ID-only
   memory in that case.
5. **Retreat** — after watering, the robot drives backward for 4 s while
   polling detections. If a new unwatered plant appears in that window,
   the bot retargets immediately. Otherwise it falls through to a scan.
6. **Two-phase scan**:
   - **Camera sweep** (PTZ only): pan +90° → 0° → -90°. If an unwatered
     plant appears at either extreme, return camera to centre and bias the
     next motor turn that way.
   - **Motor turn** (if the sweep saw nothing): rotate the whole robot
     ~180° and sweep again.
   - If the PTZ camera isn't available, the navigator skips straight to the
     original short-turn scan.

---

## 📡 API Endpoints

### Root

| Method | Endpoint | Description                   |
|--------|----------|-------------------------------|
| GET    | `/`      | System status + endpoint list |

### Motor Control

| Method | Endpoint       | Parameters                                                          | Description                  |
|--------|----------------|---------------------------------------------------------------------|------------------------------|
| POST   | `/motor/move`  | `direction` (forward/backward/left/right), `speed` (0.0–1.0)        | Drive the robot              |
| POST   | `/motor/stop`  | —                                                                   | Stop all motors immediately  |

```bash
curl -X POST "http://<pi-ip>:8000/motor/move?direction=forward&speed=0.8"
curl -X POST "http://<pi-ip>:8000/motor/stop"
```

### Camera (MJPEG)

| Method | Endpoint            | Parameters                    | Description                                       |
|--------|---------------------|-------------------------------|---------------------------------------------------|
| GET    | `/camera/feed`      | `mode` (manual \| automatic)  | Live MJPEG stream (auto adds detection overlays)  |
| GET    | `/camera/snapshot`  | —                             | Single JPEG frame                                 |

- **`manual`** — plain video stream; autonomous mode is stopped.
- **`automatic`** — overlays bounding boxes, `WATERED` / `NOT WATERED`
  labels per plant, and a `WATERING PLANT...` banner during watering.
  Starting this stream **starts the AutoNavigator**.

```html
<img src="http://<pi-ip>:8000/camera/feed" />
<img src="http://<pi-ip>:8000/camera/feed?mode=automatic" />
```

### Camera PTZ (AXIS VAPIX)

All PTZ endpoints return `503` if the AXIS camera isn't reachable.

| Method | Endpoint                  | Parameters                              | Description                                 |
|--------|---------------------------|-----------------------------------------|---------------------------------------------|
| POST   | `/camera/ptz/pan`         | `angle` (-180..180)                     | Absolute pan in degrees                      |
| POST   | `/camera/ptz/tilt`        | `angle` (-90..90)                       | Absolute tilt in degrees                     |
| POST   | `/camera/ptz/zoom`        | `level` (1..9999)                       | Absolute zoom level                          |
| POST   | `/camera/ptz/move`        | `pan_speed`, `tilt_speed` (-100..100)   | Continuous PTZ motion                        |
| POST   | `/camera/ptz/stop`        | —                                       | Stop continuous motion                       |
| POST   | `/camera/ptz/look`        | `direction` (left/right/up/down/center), `degrees` | Convenience absolute look      |
| POST   | `/camera/ptz/center`      | —                                       | Re-centre pan to 0°                          |
| POST   | `/camera/ptz/home`        | —                                       | Recall the camera's home preset              |
| POST   | `/camera/ptz/preset`      | `name` *or* `number`                    | Recall a server preset position              |
| GET    | `/camera/ptz/position`    | —                                       | Current pan / tilt / zoom values             |

```bash
# Look right 45°
curl -X POST "http://<pi-ip>:8000/camera/ptz/look?direction=right&degrees=45"

# Continuous slow pan-right until stopped
curl -X POST "http://<pi-ip>:8000/camera/ptz/move?pan_speed=20&tilt_speed=0"
curl -X POST "http://<pi-ip>:8000/camera/ptz/stop"

# Jump to a preset configured in the camera's web UI
curl -X POST "http://<pi-ip>:8000/camera/ptz/preset?name=Row_A"
```

Under the hood these call the camera at
`http://<PTZ_HOST>/axis-cgi/com/ptz.cgi` per the AXIS 213 HTTP API.

### Water Pump

| Method | Endpoint        | Description                  |
|--------|-----------------|------------------------------|
| POST   | `/pump/on`      | Turn water pump ON           |
| POST   | `/pump/off`     | Turn water pump OFF          |
| GET    | `/pump/status`  | Check if pump is running     |

### AI Detection

| Method | Endpoint      | Description                              |
|--------|---------------|------------------------------------------|
| GET    | `/ai/detect`  | Run YOLO on latest frame, return results |

**Response example:**
```json
{
  "status": "ok",
  "count": 2,
  "detections": [
    {
      "class": "plant",
      "confidence": 0.87,
      "box": [120.5, 45.2, 300.1, 280.7],
      "id": 3
    }
  ]
}
```

Each detection includes a ByteTrack `id` (or `null` if tracking is disabled
or the track is too young).

---

## 📖 API Documentation

FastAPI auto-generates interactive docs:

- **Swagger UI:** `http://<pi-ip>:8000/docs`
- **ReDoc:** `http://<pi-ip>:8000/redoc`

---

## 🛑 Shutdown

Press **Ctrl+C** to gracefully shut down. The system will:

1. Stop the AutoNavigator (if running)
2. Stop all motors (set to neutral)
3. Turn off the water pump
4. Stop the camera capture thread
5. Re-centre the PTZ camera and close its HTTP session
6. Release all GPIO pins
7. Kill the Cloudflare tunnel (if running)

---

## 📦 Dependencies

- **fastapi** / **uvicorn** – Web framework + ASGI server
- **opencv-python-headless** – Camera capture & image processing
- **ultralytics** – YOLO object detection + ByteTrack
- **torch** / **torchvision** – MobileNet-V3 backbone for appearance re-ID
  (installed as transitive deps of ultralytics; ReID disables itself if missing)
- **gpiozero** / **lgpio** – Raspberry Pi GPIO control
- **requests** – HTTP client for the AXIS VAPIX PTZ API

---

## 🌐 Cloudflare Tunnel (Optional)

For remote access outside your local network, the startup script can launch a
[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/).

**Install cloudflared on Pi:**
```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 \
  -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
```

When you run `./start.sh`, you'll be prompted to enable the tunnel. It will
print a public URL you can share.
