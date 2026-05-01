"""
Camera Controller Component
============================
Singleton camera controller that continuously captures frames
from an AXIS 213 network camera (MJPEG stream) via OpenCV.

A background thread drains the camera buffer so `get_frame()`
always returns the most recent frame with minimal latency.
"""

import logging
import threading
import time

import cv2

logger = logging.getLogger(__name__)

_DEFAULT_SOURCE = "http://169.254.138.53/axis-cgi/mjpg/video.cgi"
_RECONNECT_DELAY = 1  # seconds before reconnect attempt


class RobotCamera:
    """Thread-safe singleton camera with background capture loop."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, source: str = _DEFAULT_SOURCE):
        if self._initialized:
            return

        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.latest_frame = None
        self._frame_lock = threading.Lock()
        self.running = True
        self._initialized = True

        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="camera-capture"
        )
        self._thread.start()
        logger.info("RobotCamera initialised  (source=%s)", self.source)

    # ------------------------------------------------------------------
    # Background capture
    # ------------------------------------------------------------------

    def _capture_loop(self):
        """Continuously read frames so the buffer never goes stale."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self._frame_lock:
                    self.latest_frame = frame
            else:
                logger.warning("Camera read failed — reconnecting in %ds", _RECONNECT_DELAY)
                self.cap.release()
                time.sleep(_RECONNECT_DELAY)
                self.cap = cv2.VideoCapture(self.source)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self) -> bytes | None:
        """
        Return the latest frame as JPEG bytes, or *None* if no
        frame has been captured yet.
        """
        with self._frame_lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()

        _, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes()

    def get_raw_frame(self):
        """
        Return the latest raw OpenCV frame (numpy array), or *None*.
        Useful for passing directly to AI/vision pipelines.
        """
        with self._frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Stop the capture thread and release the camera."""
        self.running = False
        self._thread.join(timeout=3)
        self.cap.release()
        logger.info("RobotCamera closed")

        # Allow a fresh instance after close
        with self._lock:
            RobotCamera._instance = None
            self._initialized = False


# ------------------------------------------------------------------
# Quick self-test (python -m components.camera)
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    camera = RobotCamera()

    print("Waiting for first frame…")
    for _ in range(30):
        frame = camera.get_frame()
        if frame:
            print(f"✔ Got frame  ({len(frame):,} bytes)")
            break
        time.sleep(0.5)
    else:
        print("✘ No frame received after 15 s")

    camera.close()