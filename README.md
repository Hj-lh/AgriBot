# 🌱 AgriBot Backend

A **FastAPI**-powered backend for a Raspberry Pi agricultural robot. Controls motors, a water pump, a live camera feed, and AI-based plant detection — all exposed as a REST API for a frontend to connect to.

---

## 📁 Project Structure

```
Backend/
├── components/
│   ├── camera.py        # MJPEG camera stream (AXIS 213)
│   ├── ai.py            # YOLO plant/object detection
│   ├── motor.py         # Differential-drive motor control (PWM)
│   └── waterpump.py     # Relay-driven water pump
├── ai_models/
│   └── yolo11n.pt       # ← Place your YOLO model here
├── venv/                # Python virtual environment
├── main.py              # FastAPI application (entry point)
├── requirements.txt     # Python dependencies
├── start.sh             # One-command startup script
└── README.md
```

---

## ⚙️ Hardware Setup

| Component       | GPIO Pin | Notes                                      |
|-----------------|----------|--------------------------------------------|
| Motor 1 (PWM)   | GPIO 12  | Hardware PWM · R/C ESC mode (50 Hz)        |
| Motor 2 (PWM)   | GPIO 13  | Hardware PWM · mounted in reverse           |
| Water Pump Relay | GPIO 17  | Active-low relay                            |
| Camera           | Network  | AXIS 213 MJPEG at `169.254.138.53`         |

---

## 📋 System Prerequisites

Before installing Python packages, install these system-level dependencies on the Raspberry Pi:

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

## 🚀 Quick Start

### 1. Transfer to Raspberry Pi

Copy the entire `Backend/` folder to your Pi (e.g. via SCP):

```bash
scp -r Backend/ pi@<pi-ip>:~/Desktop/
```

### 2. Install system dependencies

```bash
sudo apt update
sudo apt install -y swig liblgpio-dev python3-dev python3-venv
```

### 3. Place your YOLO model

Copy `yolo11n.pt` (or your custom model) into `ai_models/`:

```bash
cp yolo11n.pt ~/Desktop/Backend/ai_models/
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

## 📡 API Endpoints

### Root

| Method | Endpoint | Description            |
|--------|----------|------------------------|
| GET    | `/`      | System status + endpoint list |

### Motor Control

| Method | Endpoint       | Parameters                          | Description                  |
|--------|----------------|-------------------------------------|------------------------------|
| POST   | `/motor/move`  | `direction` (forward/backward/left/right), `speed` (0.0–1.0) | Drive the robot |
| POST   | `/motor/stop`  | —                                   | Stop all motors immediately  |

**Example:**
```bash
# Move forward at 80% speed
curl -X POST "http://<pi-ip>:8000/motor/move?direction=forward&speed=0.8"

# Stop
curl -X POST "http://<pi-ip>:8000/motor/stop"
```

### Camera

| Method | Endpoint            | Description                     |
|--------|---------------------|---------------------------------|
| GET    | `/camera/feed`      | Live MJPEG video stream         |
| GET    | `/camera/snapshot`   | Single JPEG frame               |

**Usage in HTML:**
```html
<img src="http://<pi-ip>:8000/camera/feed" />
```

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
      "box": [120.5, 45.2, 300.1, 280.7]
    }
  ]
}
```

---

## 📖 API Documentation

FastAPI auto-generates interactive docs:

- **Swagger UI:** `http://<pi-ip>:8000/docs`
- **ReDoc:** `http://<pi-ip>:8000/redoc`

---

## 🛑 Shutdown

Press **Ctrl+C** to gracefully shut down. The system will:

1. Stop all motors (set to neutral)
2. Turn off the water pump
3. Stop the camera capture thread
4. Release all GPIO pins
5. Kill the Cloudflare tunnel (if running)

---

## 📦 Dependencies

- **fastapi** – Web framework
- **uvicorn** – ASGI server
- **opencv-python-headless** – Camera capture & image processing
- **ultralytics** – YOLO object detection
- **gpiozero** – Raspberry Pi GPIO control

---

## 🌐 Cloudflare Tunnel (Optional)

For remote access outside your local network, the startup script can launch a [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/).

**Install cloudflared on Pi:**
```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 \
  -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
```

When you run `./start.sh`, you'll be prompted to enable the tunnel. It will print a public URL you can share.
