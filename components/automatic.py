"""
Automatic Navigation Component (PID steering)
=============================================
Handles background autonomous navigation using YOLO detections.
Uses a PID controller on the horizontal offset of the plant from
the frame center to continuously steer the robot while driving
forward — smoother than discrete left/center/right zones.
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

# ---------- Speeds ----------
AUTO_BASE_FORWARD = 0.5   # forward speed when perfectly centered
AUTO_SPEED_TURN   = 0.9   # used only for scan mode (no target)

# ---------- PID gains (tune these on hardware) ----------
# error is normalized to [-1, +1] where:
#   -1 = plant at far left edge, +1 = plant at far right edge
# Start with P only. Add D if it oscillates. Only add I if it has
# a persistent steady-state offset (usually not needed here).
PID_KP = 1.2   # proportional: how hard to correct per unit of error
PID_KI = 0.0   # integral: almost never needed for steering
PID_KD = 0.3   # derivative: damps oscillation

# Anti-windup clamp for the integral term
PID_I_MAX = 0.5

# ---------- Loop timing ----------
LOOP_DT          = 0.1   # control loop period (10 Hz)
SCAN_TURN_TIME   = 0.5   # how long to spin per scan tick
SCAN_WAIT_YOLO   = 1.0   # wait after a scan burst for fresh detections
LOST_THRESHOLD   = 3     # consecutive empty frames before scan kicks in

# ---------- Geometry ----------
CENTERED_ERROR    = 0.15  # |error| below this counts as "centered enough"
ARRIVAL_AREA_RATIO = 0.3  # bbox / frame area → arrived

WATERING_DURATION = 5     # seconds


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

        # PID state
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = None

        # Lost-target tracking
        self._frames_without_target = 0
        self._scan_direction = 1  # +1 left, -1 right; flips each tick

    # ------------------------------------------------------------------
    # Public API (unchanged signatures)
    # ------------------------------------------------------------------
    def update_detections(self, detections: list[dict], frame_width: int, frame_height: int = 480):
        with self._lock:
            self.latest_detections = list(detections)
            self.frame_width = frame_width
            self.frame_height = frame_height

    def start(self):
        if self.is_active:
            return
        logger.info("AutoNavigator (PID) starting...")
        self.motor.stop()
        self.pump.off()
        self._stop_event.clear()
        self._reset_pid()
        self._frames_without_target = 0
        self.is_active = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="auto-navigator")
        self._thread.start()

    def stop(self):
        if not self.is_active:
            return
        logger.info("AutoNavigator (PID) stopping...")
        self.is_active = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self.motor.stop()
        self.pump.off()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sleep(self, duration: float) -> bool:
        """Interruptible sleep. Returns True if we should stop."""
        return self._stop_event.wait(timeout=duration)

    def _reset_pid(self):
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = None

    def _is_arrived(self, box, frame_width, frame_height) -> bool:
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_width * frame_height
        ratio = box_area / frame_area if frame_area > 0 else 0
        logger.debug("Auto: box area ratio = %.2f (threshold=%.2f)", ratio, ARRIVAL_AREA_RATIO)
        return ratio >= ARRIVAL_AREA_RATIO

    def _compute_pid(self, error: float) -> float:
        """Return steering correction in roughly [-1, +1]."""
        now = time.monotonic()
        if self._last_time is None:
            dt = LOOP_DT
        else:
            dt = max(1e-3, now - self._last_time)
        self._last_time = now

        # Integral with anti-windup clamp
        self._integral += error * dt
        self._integral = max(-PID_I_MAX, min(PID_I_MAX, self._integral))

        derivative = (error - self._last_error) / dt
        self._last_error = error

        steering = (PID_KP * error) + (PID_KI * self._integral) + (PID_KD * derivative)
        return max(-1.0, min(1.0, steering))

    def _water_and_finish(self):
        logger.info("Auto: ARRIVED at plant — watering...")
        self.motor.stop()
        for _ in range(WATERING_DURATION):
            if not self.is_active:
                break
            logger.info("watering .. watering ... watering")
            if self._sleep(1.0):
                break
        logger.info("Auto: watering complete")

    def _do_scan_tick(self) -> bool:
        """Spin a bit, then wait for fresh detections. Returns True if stop requested."""
        logger.debug("Auto: no target → scanning (direction=%d)", self._scan_direction)
        if self._scan_direction > 0:
            self.motor.left(AUTO_SPEED_TURN)
        else:
            self.motor.right(AUTO_SPEED_TURN)
        if self._sleep(SCAN_TURN_TIME):
            return True
        self.motor.stop()
        self._scan_direction *= -1  # alternate direction each tick
        if self._sleep(SCAN_WAIT_YOLO):
            return True
        # PID state is stale after a long scan break
        self._reset_pid()
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _loop(self):
        while self.is_active:
            # Snapshot latest detections
            with self._lock:
                detections = list(self.latest_detections)
                width = self.frame_width
                height = self.frame_height

            best = max(detections, key=lambda d: d["confidence"]) if detections else None

            if best is not None:
                self._frames_without_target = 0

                # Arrival check first — if we're there, stop and water
                x1, y1, x2, y2 = best["box"]
                obj_center_x = (x1 + x2) / 2.0

                # Normalized error: -1 (far left) .. 0 (centered) .. +1 (far right)
                error = ((obj_center_x / width) - 0.5) * 2.0

                if abs(error) < CENTERED_ERROR and self._is_arrived([x1, y1, x2, y2], width, height):
                    self._water_and_finish()
                    break

                # PID steering
                steering = self._compute_pid(error)

                # Differential drive mix: forward base speed + steering bias
                # steering > 0 → plant is right → turn right (slow right wheel)
                # steering < 0 → plant is left  → turn left  (slow left wheel)
                left_speed  = AUTO_BASE_FORWARD + steering
                right_speed = AUTO_BASE_FORWARD - steering

                # Clamp to motor limits
                left_speed  = max(-1.0, min(1.0, left_speed))
                right_speed = max(-1.0, min(1.0, right_speed))

                logger.debug(
                    "Auto: err=%+.2f steer=%+.2f  L=%+.2f R=%+.2f",
                    error, steering, left_speed, right_speed,
                )
                # Uses the low-level per-motor setter (motor 1 = left, motor 2 = right)
                self.motor._set_motors(left_speed, right_speed)

                if self._sleep(LOOP_DT):
                    break

            elif self._frames_without_target < LOST_THRESHOLD:
                # Brief dropout — coast to a stop, hold position, let YOLO catch up
                self._frames_without_target += 1
                logger.debug(
                    "Auto: target lost briefly (%d/%d) — holding",
                    self._frames_without_target, LOST_THRESHOLD,
                )
                self.motor.stop()
                self._reset_pid()  # don't carry stale integral/derivative
                if self._sleep(SCAN_WAIT_YOLO):
                    break

            else:
                # Genuinely lost — scan
                self._frames_without_target += 1
                if self._do_scan_tick():
                    break