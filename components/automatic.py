"""
Automatic Navigation Component
==============================
Handles background autonomous navigation using YOLO + ByteTrack detections.

Key design:
  - The navigator consumes detections as soon as they arrive via a threading
    Event, so it never misses short-lived detections.
  - ByteTrack IDs are used to remember which plants have already been
    watered so the robot does not target them a second time.
  - After watering, the robot reverses for a short window to clear the
    plant, watching for the *next* unwatered plant.
  - Scanning has two phases:
        1. **Camera sweep**  – pan the PTZ camera right 90° then left 90°
                               (no robot motion) while looking for plants.
        2. **Motor scan**    – if nothing seen, rotate the whole robot
                               ~180° and try the camera sweep again.

State machine:
  SCANNING    → target detected     → STOP + re-evaluate (no double-turn)
  TRACKING    → target in center    → APPROACHING
  TRACKING    → target left/right   → TURNING (short)
  TRACKING    → target lost briefly → HOLDING
  APPROACHING → box large enough    → WATERING
  WATERING    → done                → RETREATING (watch for next plant)
  RETREATING  → new target found    → TRACKING
  RETREATING  → nothing seen        → SCANNING (camera sweep + motor turn)
  HOLDING     → target reappears    → TRACKING
  HOLDING     → lost too long       → SCANNING
"""

import time
import threading
import logging

from components.motor import MotorController
from components.waterpump import WaterPumpController
from components.reid import PlantReID

logger = logging.getLogger(__name__)

# --- Speeds ---
AUTO_SPEED_FORWARD     = 0.6
AUTO_SPEED_BACKWARD    = 0.5
AUTO_SPEED_TURN        = 0.75
AUTO_SPEED_TURN_GENTLE = 0.375
MIN_TURN_SPEED         = 0.375

# --- Durations (how long motors run per action) ---
MOVE_TURN_DURATION    = 0.5
MOVE_FORWARD_DURATION = 1.0

# --- After stopping, wait for fresh YOLO detections ---
SLEEP_WAIT_YOLO = 1.5

# --- Scanning ---
SCAN_TURN_DURATION       = 1.0
MOTOR_180_TURN_DURATION  = 2.4    # how long to drive a roughly-180° spin
CAMERA_PAN_SETTLE        = 1.5    # seconds to wait for camera to settle + YOLO frames

# --- Camera sweep angles ---
CAM_SWEEP_RIGHT =  90.0
CAM_SWEEP_LEFT  = -90.0
CAM_SWEEP_CENTER = 0.0

# --- Stepped sweep tuning ---
# Pan in small increments and pause at each one so YOLO has clean,
# motion-blur-free frames to detect on. Tune for your camera speed.
SWEEP_STEP_DEG       = 20.0   # how far to pan between detection checks
SWEEP_STEP_PAUSE     = 1.2    # seconds to dwell at each step
SWEEP_PAN_SPEED      = 25     # slower than the default 60 for steadier frames
SWEEP_RECENTER_SPEED = 40     # slightly faster to return camera to centre

# --- Geometry ---
CENTER_MARGIN = 0.33  # Middle third of the frame is "CENTER"

# --- Arrival ---
ARRIVAL_AREA_RATIO = 0.5

# --- Watering ---
WATERING_DURATION = 5  # seconds

# --- Retreat after watering ---
RETREAT_DURATION   = 4.0   # seconds the robot drives backward
RETREAT_POLL_PERIOD = 0.1  # how often to check for new targets while reversing

# --- How many cycles with no detection before switching to scan ---
LOST_THRESHOLD = 3


class AutoNavigator:
    def __init__(
        self,
        motor: MotorController,
        pump: WaterPumpController,
        ptz=None,                     # CameraPTZController | None
        reid: PlantReID | None = None,
    ):
        self.motor = motor
        self.pump = pump
        self.ptz = ptz                # may be None if camera control unavailable
        self.reid = reid              # may be None if torchvision unavailable

        self.is_active = False
        self._thread = None

        # Detection data (written by camera stream, read by navigator)
        self.latest_detections: list[dict] = []
        self.latest_frame = None
        self.frame_width: int = 640
        self.frame_height: int = 480
        self._lock = threading.Lock()

        # Cached ReID embeddings by track ID — avoids re-running the CNN
        # on every navigation cycle for already-seen tracks.
        self._embed_cache: dict[int, "np.ndarray"] = {}

        # Signals
        self._stop_event = threading.Event()
        self._detection_event = threading.Event()

        # Watering / labelling state — read by the MJPEG annotator
        self._watered_ids: set[int] = set()
        self._is_watering: bool = False

        # Navigation state
        self._frames_without_target = 0
        self._last_known_zone: str = "LEFT"
        self._just_finished_scan: bool = False  # Prevents double-turn after scan

        # Oscillation detection
        self._zone_history: list[str] = []
        self._oscillating = False

    # ------------------------------------------------------------------
    # Public read-only accessors used by the video stream / annotator
    # ------------------------------------------------------------------

    @property
    def watered_ids(self) -> set[int]:
        return set(self._watered_ids)

    @property
    def is_watering(self) -> bool:
        return self._is_watering

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def _reset_state(self):
        """Reset all transient navigation state for a fresh run."""
        self._frames_without_target = 0
        self._last_known_zone = "LEFT"
        self._just_finished_scan = False
        self._zone_history.clear()
        self._oscillating = False
        self._is_watering = False
        self._watered_ids.clear()
        self._embed_cache.clear()
        if self.reid is not None:
            self.reid.clear()
        self._detection_event.clear()

    # ------------------------------------------------------------------
    # Called by the camera stream thread
    # ------------------------------------------------------------------

    def update_detections(
        self,
        detections: list[dict],
        frame_width: int,
        frame_height: int = 480,
        frame=None,
    ):
        """Called by the MJPEG generator every frame with YOLO results.

        ``frame`` is the raw OpenCV frame the detections came from; the
        navigator holds a reference (no copy) so it can crop watered-plant
        regions when ReID is enabled.
        """
        with self._lock:
            self.latest_detections = list(detections)
            self.frame_width = frame_width
            self.frame_height = frame_height
            if frame is not None:
                self.latest_frame = frame

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
        if self._ptz_available():
            self.ptz.look_center()
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
        self._detection_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self.motor.stop()
        self.pump.off()
        if self._ptz_available():
            self.ptz.look_center()

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
        """Wait for new detections OR timeout. Returns True if stop requested."""
        self._detection_event.clear()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
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
    # Detection filtering — ignore plants we have already watered
    # ------------------------------------------------------------------

    def _reid_available(self) -> bool:
        return self.reid is not None and self.reid.enabled

    def _embedding_for(self, det: dict, frame):
        """Return the (cached) ReID embedding for a detection, or None."""
        if not self._reid_available() or frame is None:
            return None
        track_id = det.get("id")
        if track_id is not None and track_id in self._embed_cache:
            return self._embed_cache[track_id]
        emb = self.reid.embed(frame, det.get("box"))
        if track_id is not None and emb is not None:
            # Cap the cache so it can't grow without bound during long runs
            if len(self._embed_cache) >= 256:
                self._embed_cache.pop(next(iter(self._embed_cache)))
            self._embed_cache[track_id] = emb
        return emb

    def _is_known_watered(self, det: dict, frame) -> bool:
        """Detection has been watered if its track ID is remembered, OR if
        its appearance matches a stored watered embedding."""
        track_id = det.get("id")
        if track_id is not None and track_id in self._watered_ids:
            return True
        if not self._reid_available():
            return False
        emb = self._embedding_for(det, frame)
        if emb is None:
            return False
        if self.reid.is_watered(emb):
            # Also stamp the track ID so subsequent frames hit the fast path
            if track_id is not None:
                self._watered_ids.add(track_id)
                logger.info("Auto: ReID matched watered plant — tagging id=%d", track_id)
            return True
        return False

    def _unwatered(self, detections: list[dict], frame=None) -> list[dict]:
        return [d for d in detections if not self._is_known_watered(d, frame)]

    def _snapshot_unwatered(self) -> tuple[list[dict], int, int]:
        with self._lock:
            dets = list(self.latest_detections)
            frame = self.latest_frame
            w, h = self.frame_width, self.frame_height
        return self._unwatered(dets, frame), w, h

    def _snapshot_with_frame(self) -> tuple[list[dict], "np.ndarray | None", int, int]:
        with self._lock:
            dets = list(self.latest_detections)
            frame = self.latest_frame
            w, h = self.frame_width, self.frame_height
        return dets, frame, w, h

    # ------------------------------------------------------------------
    # Phase: watering
    # ------------------------------------------------------------------

    def _do_watering(self, target: dict):
        """Run the pump for WATERING_DURATION while the annotator shows a banner.

        After the pump turns off, the watered plant's appearance embedding
        is captured (if ReID is enabled) so the plant stays recognised even
        after its ByteTrack ID is lost.
        """
        target_id = target.get("id")
        logger.info("Auto: ARRIVED at plant (id=%s) — watering...", target_id)
        self.motor.stop()
        self._is_watering = True
        self.pump.on()
        try:
            for i in range(WATERING_DURATION):
                if not self.is_active:
                    break
                logger.info("Auto: watering... (%d/%d s)", i + 1, WATERING_DURATION)
                if self._sleep(1.0):
                    break
        finally:
            self.pump.off()
            self._is_watering = False

        if target_id is not None:
            self._watered_ids.add(target_id)
            logger.info("Auto: plant id=%d marked as WATERED (total=%d)",
                        target_id, len(self._watered_ids))
        else:
            logger.info("Auto: watering complete — no track ID to remember")

        # Capture an appearance embedding so the same physical plant is
        # recognised later even after its track ID is lost.
        if self._reid_available():
            with self._lock:
                frame = self.latest_frame
            # Prefer the freshest detection of the same track ID (camera may
            # have moved slightly during watering); fall back to the target.
            box = target.get("box")
            if target_id is not None:
                with self._lock:
                    for d in self.latest_detections:
                        if d.get("id") == target_id:
                            box = d.get("box")
                            break
            emb = self.reid.embed(frame, box)
            if emb is not None:
                self.reid.add(emb)
            else:
                logger.warning("Auto: ReID embedding unavailable for watered plant")

    # ------------------------------------------------------------------
    # Phase: retreat after watering
    # ------------------------------------------------------------------

    def _retreat_and_search(self) -> bool:
        """
        Drive backward for RETREAT_DURATION while watching for any unwatered
        plant to appear in frame. Returns True if a new target showed up so
        the main loop can immediately retarget; False if the window expired
        with nothing seen.
        """
        logger.info("Auto: retreating for %.1fs to look for next plant...", RETREAT_DURATION)
        self.motor.backward(AUTO_SPEED_BACKWARD)
        deadline = time.monotonic() + RETREAT_DURATION
        found = False

        try:
            while time.monotonic() < deadline:
                if self._stop_event.is_set():
                    break
                unwatered, _, _ = self._snapshot_unwatered()
                if unwatered:
                    logger.info("Auto: new unwatered plant spotted during retreat (id=%s)",
                                unwatered[0].get("id"))
                    found = True
                    break
                if self._stop_event.wait(timeout=RETREAT_POLL_PERIOD):
                    break
        finally:
            self.motor.stop()

        return found

    # ------------------------------------------------------------------
    # Phase: scan
    # ------------------------------------------------------------------

    def _ptz_available(self) -> bool:
        return self.ptz is not None and getattr(self.ptz, "enabled", False)

    def _sweep_side(self, direction: int, total_deg: float) -> tuple[str | None, float]:
        """
        Pan the camera ``total_deg`` to one side in ``SWEEP_STEP_DEG`` steps,
        pausing at each step so YOLO can produce a clean detection.

        ``direction``  +1 = right, -1 = left.
        Returns ``(side, travelled)`` — ``side`` is "RIGHT" / "LEFT" if an
        unwatered plant was seen, else None. ``travelled`` is the unsigned
        degrees actually moved, so the caller can recentre by that amount.
        """
        side_name = "RIGHT" if direction > 0 else "LEFT"
        travelled = 0.0

        while travelled < total_deg:
            step = min(SWEEP_STEP_DEG, total_deg - travelled)
            logger.info(
                "Auto: sweep %s — step %.0f° (total %.0f°/%.0f°)",
                side_name, step, travelled + step, total_deg,
            )
            self.ptz.pan_by(direction * step, speed=SWEEP_PAN_SPEED)
            travelled += step

            # Dwell so YOLO produces motion-blur-free frames at this angle
            if self._sleep(SWEEP_STEP_PAUSE):
                return None, travelled  # stop requested

            unwatered, _, _ = self._snapshot_unwatered()
            if unwatered:
                logger.info(
                    "Auto: plant spotted on %s at sweep offset %.0f°",
                    side_name, travelled,
                )
                return side_name, travelled

        return None, travelled

    def _camera_sweep(self) -> str | None:
        """
        Pan the PTZ camera right → centre → left without moving the robot.
        Returns 'RIGHT' or 'LEFT' if an unwatered plant appears at that pan
        position, or None if nothing was seen.

        Uses *relative* timed pans because the AXIS 213 firmware doesn't
        accept absolute ``pan=<deg>`` commands — only ``continuouspantiltmove``.
        Movement is stepped (``SWEEP_STEP_DEG``) with a dwell at each step so
        the detector has clean frames to work with.
        """
        if not self._ptz_available():
            return None

        right_deg = abs(CAM_SWEEP_RIGHT)
        left_deg  = abs(CAM_SWEEP_LEFT)

        # --- Sweep right step-by-step ---
        side, travelled_right = self._sweep_side(+1, right_deg)
        if self._stop_event.is_set():
            self.ptz.pan_by(-travelled_right, speed=SWEEP_RECENTER_SPEED)
            return None
        if side is not None:
            self.ptz.pan_by(-travelled_right, speed=SWEEP_RECENTER_SPEED)
            self._sleep(0.5)
            return side

        # --- Recentre, then sweep left step-by-step ---
        logger.info("Auto: sweep RIGHT finished without a hit — recentring")
        self.ptz.pan_by(-travelled_right, speed=SWEEP_RECENTER_SPEED)
        if self._sleep(0.5):
            return None

        side, travelled_left = self._sweep_side(-1, left_deg)
        if self._stop_event.is_set():
            self.ptz.pan_by(+travelled_left, speed=SWEEP_RECENTER_SPEED)
            return None
        if side is not None:
            self.ptz.pan_by(+travelled_left, speed=SWEEP_RECENTER_SPEED)
            self._sleep(0.5)
            return side

        # --- Nothing seen — return camera to centre ---
        logger.info("Auto: sweep complete — nothing detected, recentring")
        self.ptz.pan_by(+travelled_left, speed=SWEEP_RECENTER_SPEED)
        self._sleep(0.4)
        return None

    def _motor_180(self):
        """Rotate the robot roughly 180° to the right."""
        logger.info("Auto: nothing seen — rotating robot ~180°")
        self.motor.right(AUTO_SPEED_TURN)
        self._sleep(MOTOR_180_TURN_DURATION)
        self.motor.stop()

    def _full_scan(self):
        """
        Two-phase scan:
          1) Camera sweep (right then left, no robot motion).
          2) If still nothing, rotate the robot ~180° and sweep again.
        If a plant is found, biases the next normal scan direction towards
        that side via ``_last_known_zone``.

        When the PTZ camera is unavailable the sweep is a no-op, so the scan
        gracefully degrades to the original short-turn behaviour.
        """
        side = self._camera_sweep()
        if side is None and self._ptz_available():
            self._motor_180()
            if self._stop_event.is_set():
                return
            side = self._camera_sweep()

        if side is not None:
            self._last_known_zone = side
            # Nudge the robot a short turn in that direction so the next
            # tracking iteration can fine-align with normal logic.
            logger.info("Auto: biasing scan towards %s based on sweep result", side)
            if side == "RIGHT":
                self.motor.right(AUTO_SPEED_TURN)
            else:
                self.motor.left(AUTO_SPEED_TURN)
            self._sleep(SCAN_TURN_DURATION)
            self.motor.stop()
        else:
            # Fall back to the original short-turn scan towards last-known side
            logger.info("Auto: sweep found nothing — short scan towards %s",
                        self._last_known_zone)
            if self._last_known_zone == "RIGHT":
                self.motor.right(AUTO_SPEED_TURN)
            else:
                self.motor.left(AUTO_SPEED_TURN)
            self._sleep(SCAN_TURN_DURATION)
            self.motor.stop()

        self._just_finished_scan = True
        self._frames_without_target = 0
        self._wait_for_detections(SLEEP_WAIT_YOLO)

    # ------------------------------------------------------------------
    # Main navigation loop
    # ------------------------------------------------------------------

    def _loop(self):
        try:
            while self.is_active:
                # 1. Snapshot latest unwatered detections
                detections, width, height = self._snapshot_unwatered()

                # 2. Pick best detection
                best = max(detections, key=lambda d: d["confidence"]) if detections else None

                # 3. Act on detection
                if best is not None:
                    self._frames_without_target = 0
                    x1, y1, x2, y2 = best["box"]
                    obj_center_x = (x1 + x2) / 2.0
                    zone = self._get_zone(obj_center_x, width)

                    # Track which side the plant was last seen on, for scan bias
                    if obj_center_x < width / 2.0:
                        self._last_known_zone = "LEFT"
                    else:
                        self._last_known_zone = "RIGHT"

                    self._update_oscillation(zone)
                    turn_speed = self._get_turn_speed()

                    # If we just finished a scan, settle before stacking a turn
                    if self._just_finished_scan:
                        logger.info("Auto: target acquired after scan — stopping to re-evaluate")
                        self._just_finished_scan = False
                        self.motor.stop()
                        if self._wait_for_detections(SLEEP_WAIT_YOLO):
                            break
                        continue

                    # --- ARRIVED: water the plant ---
                    if zone == "CENTER" and self._is_arrived([x1, y1, x2, y2], width, height):
                        self._do_watering(best)
                        if not self.is_active:
                            break

                        # Retreat and look for the next unwatered plant
                        if self._retreat_and_search():
                            # New target spotted — go straight back into the
                            # loop, which will re-acquire on the next pass.
                            continue
                        if self._stop_event.is_set():
                            break

                        # Nothing seen during retreat — fall through to scan
                        self._full_scan()
                        continue

                    # --- TURN LEFT ---
                    if zone == "LEFT":
                        logger.debug("Auto: plant LEFT → turning left (speed=%.2f, %.1fs)",
                                     turn_speed, MOVE_TURN_DURATION)
                        self.motor.left(turn_speed)
                        if self._sleep(MOVE_TURN_DURATION):
                            break

                    # --- TURN RIGHT ---
                    elif zone == "RIGHT":
                        logger.debug("Auto: plant RIGHT → turning right (speed=%.2f, %.1fs)",
                                     turn_speed, MOVE_TURN_DURATION)
                        self.motor.right(turn_speed)
                        if self._sleep(MOVE_TURN_DURATION):
                            break

                    # --- CENTER but not arrived → approach ---
                    else:
                        logger.debug("Auto: plant CENTER → forward")
                        self.motor.forward(AUTO_SPEED_FORWARD)
                        if self._sleep(MOVE_FORWARD_DURATION):
                            break

                    self.motor.stop()
                    if self._wait_for_detections(SLEEP_WAIT_YOLO):
                        break

                # 4. Target lost briefly — hold and wait
                elif self._frames_without_target < LOST_THRESHOLD:
                    self._frames_without_target += 1
                    logger.debug("Auto: target lost briefly (%d/%d) — holding",
                                 self._frames_without_target, LOST_THRESHOLD)
                    self.motor.stop()
                    if self._wait_for_detections(SLEEP_WAIT_YOLO):
                        break

                # 5. Target genuinely lost — full two-phase scan
                else:
                    self._full_scan()
                    if self._stop_event.is_set():
                        break

        finally:
            self.motor.stop()
            self.pump.off()
            self._is_watering = False
            if self._ptz_available():
                try:
                    self.ptz.look_center()
                except Exception:  # noqa: BLE001
                    pass
            self.is_active = False
            logger.info("AutoNavigator loop exited — cleanup complete")
