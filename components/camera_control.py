"""
Camera PTZ Control Component
============================
HTTP-based pan / tilt / zoom controller for the AXIS 213 PTZ
network camera using the VAPIX (``axis-cgi``) API.

The AXIS 213 exposes PTZ commands over HTTP at:

    http://<camera-host>/axis-cgi/com/ptz.cgi?<query>

Examples
--------
    pan to +45° absolute :   ?camera=1&pan=45
    relative pan +10°    :   ?camera=1&rpan=10
    tilt to -20°         :   ?camera=1&tilt=-20
    set zoom (1..9999)   :   ?camera=1&zoom=2000
    continuous move      :   ?camera=1&continuouspantiltmove=50,0
    stop continuous move :   ?camera=1&continuouspantiltmove=0,0
    go to home preset    :   ?camera=1&move=home
    query position       :   ?query=position
"""

import logging
import threading
import time

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

_DEFAULT_HOST     = "169.254.138.53"
_DEFAULT_USER     = "root"
_DEFAULT_PASSWORD = "root"
_DEFAULT_TIMEOUT  = 3.0


class CameraPTZController:
    """High-level PTZ controller for the AXIS 213 via VAPIX over HTTP."""

    def __init__(
        self,
        host: str = _DEFAULT_HOST,
        user: str = _DEFAULT_USER,
        password: str = _DEFAULT_PASSWORD,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self.host = host
        self.timeout = timeout
        self.base_url = f"http://{host}/axis-cgi/com/ptz.cgi"

        self.session = requests.Session()
        self.session.auth = (user, password)  # AXIS 213 accepts Basic auth
        # The AXIS 213 has a tiny HTTP socket budget — keep-alive sessions
        # exhaust it within a few PTZ calls and the next request fails with
        # urllib3 ``HTTPConnectionPool: Max retries exceeded``. Force a fresh
        # single connection per request and let the camera close it.
        self.session.headers.update({"Connection": "close"})
        adapter = HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=0,
            pool_block=True,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._lock = threading.Lock()
        self.enabled = False  # flipped True only if the probe succeeds

        # Probe once so we know early if the camera is reachable. If it isn't,
        # leave ``enabled`` False so callers (notably the AutoNavigator) skip
        # PTZ-dependent code paths instead of blocking on every timeout.
        try:
            if self.get_position() is not None:
                self.enabled = True
                logger.info("CameraPTZController initialised  (host=%s)", host)
            else:
                logger.warning(
                    "PTZ probe returned no data — control disabled until reachable"
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("PTZ probe failed (%s) — control disabled until reachable", e)

    # ------------------------------------------------------------------
    # Low-level transport
    # ------------------------------------------------------------------

    def _send(self, params: dict) -> bool:
        """Send a VAPIX request. Returns True on HTTP 200 **and** a non-error
        body, False otherwise.

        AXIS cameras frequently return HTTP 200 for unsupported commands and
        signal the failure only via an ``# Error: ...`` line in the body —
        which is exactly the silent-failure mode the AXIS 213 exhibits when
        sent newer VAPIX-2 absolute commands like ``pan=<deg>``.
        """
        if not self.enabled:
            return False
        try:
            with self._lock:
                resp = self.session.get(
                    self.base_url, params=params, timeout=self.timeout
                )
            body = (resp.text or "").strip()
            ok_status = 200 <= resp.status_code < 300
            looks_like_error = body.lower().startswith(("error", "# error"))
            if ok_status and not looks_like_error:
                logger.info("PTZ ok %s | %s", params, body[:80] or "<empty>")
                return True
            logger.warning(
                "PTZ %s -> HTTP %d (%s): %s",
                params, resp.status_code,
                "body-error" if looks_like_error else "http-error",
                body[:160] or "<empty body>",
            )
            return False
        except requests.exceptions.ConnectionError as e:
            logger.warning(
                "PTZ connection error (camera busy / closed socket): %s",
                str(e).split("\n", 1)[0][:160],
            )
            return False
        except requests.exceptions.Timeout:
            logger.warning("PTZ request timed out after %.1fs", self.timeout)
            return False
        except Exception as e:  # noqa: BLE001
            logger.error("PTZ request error: %s", e)
            return False

    # ------------------------------------------------------------------
    # Absolute movement
    # ------------------------------------------------------------------

    def pan(self, angle: float) -> bool:
        """Absolute pan in degrees (typically -180..180)."""
        return self._send({"camera": 1, "pan": angle})

    def tilt(self, angle: float) -> bool:
        """Absolute tilt in degrees (typically -90..90)."""
        return self._send({"camera": 1, "tilt": angle})

    def pan_tilt(self, pan: float, tilt: float) -> bool:
        return self._send({"camera": 1, "pan": pan, "tilt": tilt})

    def zoom(self, level: int) -> bool:
        """Absolute zoom (1 = wide, 9999 = telephoto)."""
        level = max(1, min(9999, int(level)))
        return self._send({"camera": 1, "zoom": level})

    # ------------------------------------------------------------------
    # Relative movement
    # ------------------------------------------------------------------

    def pan_relative(self, delta: float) -> bool:
        return self._send({"camera": 1, "rpan": delta})

    def tilt_relative(self, delta: float) -> bool:
        return self._send({"camera": 1, "rtilt": delta})

    def zoom_relative(self, delta: int) -> bool:
        return self._send({"camera": 1, "rzoom": int(delta)})

    # ------------------------------------------------------------------
    # Continuous movement
    # ------------------------------------------------------------------

    def move_continuous(self, pan_speed: int, tilt_speed: int) -> bool:
        """Pan/tilt speeds in range -100..100. (0, 0) stops motion."""
        pan_speed  = max(-100, min(100, int(pan_speed)))
        tilt_speed = max(-100, min(100, int(tilt_speed)))
        return self._send(
            {"camera": 1, "continuouspantiltmove": f"{pan_speed},{tilt_speed}"}
        )

    def stop_movement(self) -> bool:
        return self._send({"camera": 1, "continuouspantiltmove": "0,0"})

    def move_for(self, duration: float, pan_speed: int = 0, tilt_speed: int = 0) -> bool:
        """Continuous pan/tilt for ``duration`` seconds, then auto-stop.

        This is the AXIS-213-friendly equivalent of ``pan(<deg>)`` because
        the 213's firmware doesn't accept absolute pan/tilt commands — only
        ``continuouspantiltmove``. Speeds are in the range -100..100.
        """
        if not self.move_continuous(pan_speed, tilt_speed):
            return False
        try:
            time.sleep(max(0.0, duration))
        finally:
            self.stop_movement()
        return True

    # Calibration: roughly how long the 213 takes to sweep 1° at speed 100.
    # Re-tune by eye if your firmware/move-speed setting differs.
    DEG_PER_SEC_AT_FULL = 60.0

    def pan_by(self, degrees: float, speed: int = 60) -> bool:
        """Pan by a *relative* angle using a timed continuous move.

        Positive = right, negative = left. Use this on the AXIS 213
        instead of :meth:`pan`, which the 213 firmware does not support.
        """
        speed = max(1, min(100, abs(int(speed))))
        sign = 1 if degrees > 0 else -1
        secs = abs(degrees) / (self.DEG_PER_SEC_AT_FULL * (speed / 100.0))
        return self.move_for(secs, pan_speed=sign * speed, tilt_speed=0)

    def tilt_by(self, degrees: float, speed: int = 60) -> bool:
        """Tilt by a relative angle using a timed continuous move."""
        speed = max(1, min(100, abs(int(speed))))
        sign = 1 if degrees > 0 else -1
        secs = abs(degrees) / (self.DEG_PER_SEC_AT_FULL * (speed / 100.0))
        return self.move_for(secs, pan_speed=0, tilt_speed=sign * speed)

    # ------------------------------------------------------------------
    # Convenience presets
    # ------------------------------------------------------------------

    def home(self) -> bool:
        """Recall the camera's configured home preset."""
        return self._send({"camera": 1, "move": "home"})

    def goto_preset(self, name: str) -> bool:
        """Recall a named server preset configured via the camera web UI."""
        return self._send({"camera": 1, "gotoserverpresetname": name})

    def goto_preset_number(self, number: int) -> bool:
        """Recall a server preset by its numeric index (1-based)."""
        return self._send({"camera": 1, "gotoserverpresetno": int(number)})

    def look_center(self) -> bool:
        return self.pan(0)

    def look_right(self, degrees: float = 90.0) -> bool:
        return self.pan(abs(degrees))

    def look_left(self, degrees: float = 90.0) -> bool:
        return self.pan(-abs(degrees))

    def look_up(self, degrees: float = 30.0) -> bool:
        return self.tilt(abs(degrees))

    def look_down(self, degrees: float = 30.0) -> bool:
        return self.tilt(-abs(degrees))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_position(self) -> dict | None:
        """Return {pan, tilt, zoom, ...} or None on failure."""
        try:
            with self._lock:
                resp = self.session.get(
                    self.base_url,
                    params={"query": "position"},
                    timeout=self.timeout,
                )
            if resp.status_code != 200:
                return None
            data: dict = {}
            for line in resp.text.strip().splitlines():
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                try:
                    data[k] = float(v)
                except ValueError:
                    data[k] = v
            return data
        except Exception as e:  # noqa: BLE001
            logger.error("PTZ query error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        try:
            self.stop_movement()
        except Exception:  # noqa: BLE001
            pass
        self.session.close()
        logger.info("CameraPTZController closed")


# ----------------------------------------------------------------------
# Quick self-test (python -m components.camera_control)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ptz = CameraPTZController()

    print("Position:", ptz.get_position())
    print("Look right 90°…")
    ptz.look_right(90)
    time.sleep(2)
    print("Center…")
    ptz.look_center()
    time.sleep(2)
    print("Look left 90°…")
    ptz.look_left(90)
    time.sleep(2)
    print("Center…")
    ptz.look_center()

    ptz.close()
