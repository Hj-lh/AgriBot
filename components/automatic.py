"""
Automatic Navigation Component
==============================
Handles background autonomous navigation using YOLO detections.

Key design:
  - The navigator consumes detections as soon as they arrive via a threading Event,
    so it never misses short-lived detections.
  - Scan direction remembers where the target was last seen (left/right).

State machine:
  SCANNING  → target detected       → TRACKING
  TRACKING  → target in center      → APPROACHING
  TRACKING  → target left/right     → TURNING
  TRACKING  → target lost briefly   → HOLDING
  APPROACHING → box large enough    → WATERING
  HOLDING   → target reappears      → TRACKING
  HOLDING   → lost too long         → SCANNING
  WATERING  → done                  → IDLE (loop exits)
"""

import time
import threading
import logging

from components.motor import MotorController
from components.waterpump import WaterPumpController

logger = logging.getLogger(__name__)

# --- Speeds ---
AUTO_SPEED_FORWARD     = 0.6
AUTO_SPEED_TURN        = 0.75
AUTO_SPEED_TURN_GENTLE = 0.25
MIN_TURN_SPEED         = 0.25

# --- Durations (how long motors run per action) ---
MOVE_TURN_DURATION    = 2
MOVE_FORWARD_DURATION = 1.0

# --- After stopping, wait for fresh YOLO detections ---
# Reduced from 2.0 → 0.5 so the navigator reacts much faster
SLEEP_WAIT_YOLO = 0.5

# --- Scanning ---
SCAN_TURN_DURATION = 2

# --- Geometry ---
CENTER_MARGIN = 0.33  # Middle third of the frame

# --- Arrival ---
ARRIVAL_AREA_RATIO = 0.5

# --- Watering ---
WATERING_DURATION = 5  # seconds

# --- How many cycles with no detection before switching to scan ---
LOST_THRESHOLD = 3


class AutoNavigator:
    def __init__(self, motor: MotorController, pump: WaterPumpController):
        self.motor = motor
        self.pump = pump

        self.is_active = False
        self._thread = None

        # Detection data (written by camera stream, read by navigator)
        self.latest_detections: list[dict] = []
        self.frame_width: int = 640
        self.frame_height: int = 480
        self._lock = threading.Lock()

        # Signals
        self._stop_event = threading.Event()
        self._detection_event = threading.Event()  # NEW: wakes navigator when detections arrive

        # Navigation state
        self._frames_without_target = 0
        self._last_known_zone: str = "LEFT"  # NEW: remember where target was last seen

        # Oscillation detection
        self._zone_history: list[str] = []
        self._oscillating = False

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def _reset_state(self):
        """Reset all transient navigation state for a fresh run."""
        self._frames_without_target = 0
        self._last_known_zone = "LEFT"
        self._zone_history.clear()
        self._oscillating = False
        self._detection_event.clear()

    # ------------------------------------------------------------------
    # Called by the camera stream thread
    # ------------------------------------------------------------------

    def update_detections(self, detections: list[dict], frame_width: int, frame_height: int = 480):
        """Called by the MJPEG generator every frame with YOLO results."""
        with self._lock:
            self.latest_detections = list(detections)
            self.frame_width = frame_width
            self.frame_height = frame_height

        # Wake the navigator immediately if there are detections
        if detections:
            self._detection_event.set()

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self):
        if self.is_active:
            return
        logger.info("AutoNavigator starting...")
        self.motor.stop()
        self.pump.off()
        self._reset_state()
        self._stop_event.clear()
        self.is_active = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="auto-navigator")
        self._thread.start()

    def stop(self):
        if not self.is_active:
            return
        logger.info("AutoNavigator stopping...")
        self.is_active = False
        self._stop_event.set()
        self._detection_event.set()  # Wake from any wait
        if self._thread:
            self._thread.join(timeout=3.0)
        self.motor.stop()
        self.pump.off()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_zone(self, obj_center_x: float, width: int) -> str:
        if obj_center_x < width * CENTER_MARGIN:
            return "LEFT"
        if obj_center_x > width * (1 - CENTER_MARGIN):
            return "RIGHT"
        return "CENTER"

    def _sleep(self, duration: float) -> bool:
        """Interruptible sleep. Returns True if we should stop."""
        return self._stop_event.wait(timeout=duration)

    def _wait_for_detections(self, timeout: float) -> bool:
        """
        Wait until new detections arrive OR timeout.
        Returns True if we should stop (stop_event fired).
        
        This replaces the old blind sleep — the navigator wakes up
        immediately when YOLO produces detections instead of sleeping
        through them.
        """
        # Wait for either: detections arrive, or stop requested, or timeout
        self._detection_event.clear()
        
        # Use a loop to check stop_event while waiting for detections
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Wait for detection event with short timeout so we can check stop
            if self._detection_event.wait(timeout=min(remaining, 0.1)):
                self._detection_event.clear()
                break
        
        return self._stop_event.is_set()

    def _is_arrived(self, box: list[float], frame_width: int, frame_height: int) -> bool:
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_width * frame_height
        ratio = box_area / frame_area if frame_area > 0 else 0
        logger.debug("Auto: box area ratio = %.2f (threshold=%.2f)", ratio, ARRIVAL_AREA_RATIO)
        return ratio >= ARRIVAL_AREA_RATIO

    def _get_turn_speed(self) -> float:
        speed = AUTO_SPEED_TURN_GENTLE if self._oscillating else AUTO_SPEED_TURN
        return max(speed, MIN_TURN_SPEED)

    def _update_oscillation(self, zone: str):
        """Track zone history and detect L↔R oscillation patterns."""
        self._zone_history.append(zone)
        if len(self._zone_history) > 4:
            self._zone_history.pop(0)

        if zone == "CENTER":
            if self._oscillating:
                logger.info("Auto: plant centered — resuming normal speed")
            self._oscillating = False
            self._zone_history.clear()
            return

        if len(self._zone_history) >= 3:
            recent = self._zone_history[-3:]
            if all(z == recent[0] for z in recent):
                if self._oscillating:
                    logger.info("Auto: zone stabilised (%s) — resuming normal speed", recent[0])
                self._oscillating = False
                return

            a, b, c = recent
            if a == c and a != b and a in ("LEFT", "RIGHT") and b in ("LEFT", "RIGHT"):
                if not self._oscillating:
                    logger.info("Auto: oscillation detected (%s→%s→%s) — using gentle speed", a, b, c)
                self._oscillating = True

    # ------------------------------------------------------------------
    # Main navigation loop
    # ------------------------------------------------------------------

    def _loop(self):
        try:
            while self.is_active:
                # 1. Snapshot latest detections
                with self._lock:
                    detections = list(self.latest_detections)
                    width = self.frame_width
                    height = self.frame_height

                # 2. Pick best detection
                best = max(detections, key=lambda d: d["confidence"]) if detections else None

                # 3. Act on detection
                if best is not None:
                    self._frames_without_target = 0
                    x1, y1, x2, y2 = best["box"]
                    obj_center_x = (x1 + x2) / 2.0
                    zone = self._get_zone(obj_center_x, width)

                    # Remember where we last saw the target (for smart scanning)
                    if zone != "CENTER":
                        self._last_known_zone = zone

                    self._update_oscillation(zone)
                    turn_speed = self._get_turn_speed()

                    # --- ARRIVED: water the plant ---
                    if zone == "CENTER" and self._is_arrived([x1, y1, x2, y2], width, height):
                        logger.info("Auto: ARRIVED at plant — watering...")
                        self.motor.stop()
                        self.pump.on()
                        for i in range(WATERING_DURATION):
                            if not self.is_active:
                                break
                            logger.info("Auto: watering... (%d/%d s)", i + 1, WATERING_DURATION)
                            if self._sleep(1.0):
                                break
                        self.pump.off()
                        logger.info("Auto: watering complete")
                        break

                    # --- TURN LEFT ---
                    if zone == "LEFT":
                        logger.debug("Auto: plant LEFT → turning left (speed=%.2f)", turn_speed)
                        self.motor.left(turn_speed)
                        if self._sleep(MOVE_TURN_DURATION):
                            break

                    # --- TURN RIGHT ---
                    elif zone == "RIGHT":
                        logger.debug("Auto: plant RIGHT → turning right (speed=%.2f)", turn_speed)
                        self.motor.right(turn_speed)
                        if self._sleep(MOVE_TURN_DURATION):
                            break

                    # --- CENTER but not arrived → approach ---
                    else:
                        logger.debug("Auto: plant CENTER → forward")
                        self.motor.forward(AUTO_SPEED_FORWARD)
                        if self._sleep(MOVE_FORWARD_DURATION):
                            break

                    # Stop motors and wait for fresh YOLO frame
                    self.motor.stop()
                    if self._wait_for_detections(SLEEP_WAIT_YOLO):
                        break

                # 4. Target lost briefly — hold position and wait for detections
                elif self._frames_without_target < LOST_THRESHOLD:
                    self._frames_without_target += 1
                    logger.debug("Auto: target lost briefly (%d/%d) — holding",
                                 self._frames_without_target, LOST_THRESHOLD)
                    self.motor.stop()
                    # Wait for detections to arrive — wake up immediately if they do
                    if self._wait_for_detections(SLEEP_WAIT_YOLO):
                        break

                # 5. Target genuinely lost — scan in the direction we last saw it
                else:
                    self._frames_without_target += 1
                    scan_direction = self._last_known_zone  # "LEFT" or "RIGHT"

                    if scan_direction == "RIGHT":
                        logger.debug("Auto: no target → scanning RIGHT (last seen right)")
                        self.motor.right(AUTO_SPEED_TURN)
                    else:
                        logger.debug("Auto: no target → scanning LEFT (last seen left)")
                        self.motor.left(AUTO_SPEED_TURN)

                    if self._sleep(SCAN_TURN_DURATION):
                        break

                    # Stop and wait for fresh detections
                    self.motor.stop()
                    if self._wait_for_detections(SLEEP_WAIT_YOLO):
                        break

        finally:
            self.motor.stop()
            self.pump.off()
            self.is_active = False
            logger.info("AutoNavigator loop exited — cleanup complete")
