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

# Load .env settings
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

AUTO_SPEED_FORWARD = float(os.getenv("AUTO_SPEED_FORWARD", "0.8"))
AUTO_SPEED_BACKWARD = float(os.getenv("AUTO_SPEED_BACKWARD", "0.8"))
AUTO_SPEED_TURN = float(os.getenv("AUTO_SPEED_TURN", "0.6"))

SLEEP_TURN_DURATION = float(os.getenv("SLEEP_TURN_DURATION", "0.5"))
SLEEP_FORWARD_DURATION = float(os.getenv("SLEEP_FORWARD_DURATION", "2.0"))
SLEEP_BACKWARD_DURATION = float(os.getenv("SLEEP_BACKWARD_DURATION", "1.0"))
SLEEP_WAIT_YOLO = float(os.getenv("SLEEP_WAIT_YOLO", "1.0"))

CENTER_MARGIN = 0.33  # Middle third


class AutoNavigator:
    def __init__(self, motor: MotorController, pump: WaterPumpController):
        self.motor = motor
        self.pump = pump
        
        self.is_active = False
        self._thread = None
        
        # We store the latest detection state updated by the camera stream
        self.latest_detections = []
        self.frame_width = 640 # Default fallback, will be updated by stream
        self._lock = threading.Lock()

    def update_detections(self, detections: list[dict], frame_width: int):
        """Called constantly by the camera stream."""
        with self._lock:
            self.latest_detections = detections
            self.frame_width = frame_width

    def start(self):
        """Start the background navigation loop."""
        if self.is_active:
            return
            
        logger.info("AutoNavigator starting...")
        self.motor.stop()
        self.pump.off()
        
        self.is_active = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="auto-navigator")
        self._thread.start()

    def stop(self):
        """Stop the background navigation loop."""
        if not self.is_active:
            return
            
        logger.info("AutoNavigator stopping...")
        self.is_active = False
        if self._thread:
            self._thread.join(timeout=2.0)
        
        # Reset hardware safely
        self.motor.stop()
        self.pump.off()

    def _get_direction(self, obj_center_x: float, width: int) -> str:
        left_boundary = width * CENTER_MARGIN
        right_boundary = width * (1 - CENTER_MARGIN)

        if obj_center_x < left_boundary:
            return "LEFT"
        elif obj_center_x > right_boundary:
            return "RIGHT"
        else:
            return "CENTER"

    def _loop(self):
        """Main navigation loop."""
        while self.is_active:
            # 1. Grab thread-safe copy of latest detections
            with self._lock:
                detections = self.latest_detections
                width = self.frame_width

            # 2. Logic: Find highest confidence object (target_class filtering is done by ai.py)
            best_det = None
            best_conf = 0
            for det in detections:
                if det["confidence"] > best_conf:
                    best_conf = det["confidence"]
                    best_det = det
            
            # 3. Handle Movement based on object
            if best_det:
                x1, y1, x2, y2 = best_det["box"]
                obj_center_x = (x1 + x2) / 2
                direction = self._get_direction(obj_center_x, width)
                
                if direction == "LEFT":
                    logger.debug("Auto: Target LEFT -> Turning left")
                    self.motor.left(AUTO_SPEED_TURN)
                    time.sleep(SLEEP_TURN_DURATION)
                    
                elif direction == "RIGHT":
                    logger.debug("Auto: Target RIGHT -> Turning right")
                    self.motor.right(AUTO_SPEED_TURN)
                    time.sleep(SLEEP_TURN_DURATION)
                    
                else: # CENTER
                    logger.debug("Auto: Target CENTERED -> Moving forward")
                    self.motor.forward(AUTO_SPEED_FORWARD)
                    time.sleep(SLEEP_FORWARD_DURATION)
                    
                # Always stop and wait after a movement to let YOLO recalculate
                self.motor.stop()
                if self.is_active:
                    time.sleep(SLEEP_WAIT_YOLO)

            else:
                # No target found
                logger.debug("Auto: No target -> Reversing to search")
                self.motor.backward(AUTO_SPEED_BACKWARD)
                time.sleep(SLEEP_BACKWARD_DURATION)
                self.motor.stop()
                
                if self.is_active:
                    # Give it time to see if target entered frame
                    time.sleep(SLEEP_WAIT_YOLO)
