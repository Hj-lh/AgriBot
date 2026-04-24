"""
Automatic Navigation Component
==============================
Handles background autonomous navigation using YOLO detections.
"""

import os
import time
import threading
import logging
from dotenv import load_dotenv

from components.motor import MotorController
from components.waterpump import WaterPumpController

logger = logging.getLogger(__name__)

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

AUTO_SPEED_FORWARD = 0.6
AUTO_SPEED_TURN    = 0.9

# How long to actually run the motors — keep turns SHORT so the robot
# doesn't overshoot. At 0.9 speed, 0.4s is roughly a ~20-30 degree pivot.
MOVE_TURN_DURATION    = 0.4
MOVE_FORWARD_DURATION = 1

# After stopping, wait for YOLO to produce a fresh detection on the new view.
# With 3–8 FPS, 1.5s gives at least 4–12 new frames to process.
SLEEP_WAIT_YOLO = 2

# How long to spin per scan tick when no target is visible
SCAN_TURN_DURATION = 0.5

CENTER_MARGIN = 0.33  # Middle third of the frame

# When the plant's bounding box fills this fraction of the frame, we've arrived.
# Tune this by testing — depends on camera FOV and desired stopping distance.
ARRIVAL_AREA_RATIO = 0.3

WATERING_DURATION = 5  # seconds to "water" (just logging for now)


class AutoNavigator:
    def __init__(self, motor: MotorController, pump: WaterPumpController):
        self.motor = motor
        self.pump = pump

        self.is_active = False
        self._thread = None

        self.latest_detections: list[dict] = []
        self.frame_width: int = 640
        self.frame_height: int = 480
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._scan_direction = 1  # +1 = left, -1 = right; flips each no-target tick

    def update_detections(self, detections: list[dict], frame_width: int, frame_height: int = 480):
        """Called constantly by the camera stream."""
        with self._lock:
            self.latest_detections = list(detections)  # snapshot, not reference
            self.frame_width = frame_width
            self.frame_height = frame_height

    def start(self):
        if self.is_active:
            return
        logger.info("AutoNavigator starting...")
        self.motor.stop()
        self.pump.off()
        self._stop_event.clear()
        self.is_active = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="auto-navigator")
        self._thread.start()

    def stop(self):
        if not self.is_active:
            return
        logger.info("AutoNavigator stopping...")
        self.is_active = False
        self._stop_event.set()          # instantly wake the thread from any sleep
        if self._thread:
            self._thread.join(timeout=3.0)
        self.motor.stop()
        self.pump.off()

    def _get_zone(self, obj_center_x: float, width: int) -> str:
        if obj_center_x < width * CENTER_MARGIN:
            return "LEFT"
        if obj_center_x > width * (1 - CENTER_MARGIN):
            return "RIGHT"
        return "CENTER"

    def _sleep(self, duration: float) -> bool:
        """Interruptible sleep. Returns True if we should stop."""
        return self._stop_event.wait(timeout=duration)

    def _is_arrived(self, box: list[float], frame_width: int, frame_height: int) -> bool:
        """Check if the plant's bounding box is large enough to consider arrived."""
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_width * frame_height
        ratio = box_area / frame_area if frame_area > 0 else 0
        logger.debug("Auto: box area ratio = %.2f (threshold=%.2f)", ratio, ARRIVAL_AREA_RATIO)
        return ratio >= ARRIVAL_AREA_RATIO

    def _loop(self):
        while self.is_active:
            # 1. Snapshot latest detections
            with self._lock:
                detections = list(self.latest_detections)
                width = self.frame_width
                height = self.frame_height

            # 2. Pick best detection
            best = max(detections, key=lambda d: d["confidence"]) if detections else None

            # 3. Move
            if best is not None:
                x1, y1, x2, y2 = best["box"]
                obj_center_x = (x1 + x2) / 2.0
                zone = self._get_zone(obj_center_x, width)

                # Check if we've arrived (box is large enough)
                if zone == "CENTER" and self._is_arrived([x1, y1, x2, y2], width, height):
                    logger.info("Auto: ARRIVED at plant — watering...")
                    self.motor.stop()
                    # Simulate watering with logging
                    for i in range(WATERING_DURATION):
                        if not self.is_active:
                            break
                        logger.info("watering .. watering ... watering")
                        if self._sleep(1.0):
                            break
                    logger.info("Auto: watering complete")
                    break  # stop automatic mode after watering

                if zone == "LEFT":
                    logger.debug("Auto: plant LEFT → turning left (%.2fs)", MOVE_TURN_DURATION)
                    self.motor.left(AUTO_SPEED_TURN)
                    if self._sleep(MOVE_TURN_DURATION):
                        break

                elif zone == "RIGHT":
                    logger.debug("Auto: plant RIGHT → turning right (%.2fs)", MOVE_TURN_DURATION)
                    self.motor.right(AUTO_SPEED_TURN)
                    if self._sleep(MOVE_TURN_DURATION):
                        break

                else:  # CENTER — not arrived yet, keep approaching
                    logger.debug("Auto: plant CENTER → forward (%.2fs)", MOVE_FORWARD_DURATION)
                    self.motor.forward(AUTO_SPEED_FORWARD)
                    if self._sleep(MOVE_FORWARD_DURATION):
                        break

            else:
                # No target — alternate left/right each tick to scan
                logger.debug("Auto: no target → scanning (direction=%d)", self._scan_direction)
                if self._scan_direction > 0:
                    self.motor.left(AUTO_SPEED_TURN)
                else:
                    self.motor.right(AUTO_SPEED_TURN)
                if self._sleep(SCAN_TURN_DURATION):
                    break
                # self._scan_direction *= -1  # flip for next no-target tick

            # 4. Stop and wait for YOLO to catch up
            self.motor.stop()
            if self.is_active:
                if self._sleep(SLEEP_WAIT_YOLO):
                    break