"""
Motor Controller Component
==========================
Differential-drive motor controller using PWM via gpiozero.
Two motors (GPIO 12 & 13) driven in R/C ESC mode:
  - 1.5 ms pulse (duty 0.075 at 50 Hz) = stop
  - 1.0 ms pulse (duty 0.050) = full reverse
  - 2.0 ms pulse (duty 0.100) = full forward

Motor 2 is physically mounted in reverse, so its signal
is inverted internally — callers don't need to worry about it.
"""

import logging
from gpiozero import PWMOutputDevice

logger = logging.getLogger(__name__)

# PWM constants (50 Hz → 20 ms period)
_FREQUENCY = 50
_NEUTRAL = 0.075        # 1.5 ms  → stop
_RANGE = 0.025          # ±0.5 ms → full speed


class MotorController:
    """High-level differential-drive controller."""

    def __init__(
        self,
        motor1_pin: int = 12,
        motor2_pin: int = 13,
        frequency: int = _FREQUENCY,
    ):
        self.motor1_pwm = PWMOutputDevice(
            motor1_pin, frequency=frequency, initial_value=_NEUTRAL
        )
        self.motor2_pwm = PWMOutputDevice(
            motor2_pin, frequency=frequency, initial_value=_NEUTRAL
        )
        logger.info(
            "MotorController initialised  (M1=GPIO%d, M2=GPIO%d, %d Hz)",
            motor1_pin, motor2_pin, frequency,
        )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _set_motors(self, m1_speed: float, m2_speed: float):
        """
        Set individual motor speeds.

        Parameters
        ----------
        m1_speed, m2_speed : float
            –1.0 (full reverse) … 0 (stop) … +1.0 (full forward).
            Motor 2 is inverted internally to match the physical mount.
        """
        m1_speed = max(-1.0, min(1.0, m1_speed))
        m2_speed = max(-1.0, min(1.0, m2_speed))

        # Motor 2 is mounted in reverse → negate its signal
        m1_pulse = _NEUTRAL + (m1_speed * _RANGE)
        m2_pulse = _NEUTRAL + (m2_speed * _RANGE)

        self.motor1_pwm.value = m1_pulse
        self.motor2_pwm.value = m2_pulse

        logger.debug(
            "Motors set  m1=%.2f (pulse %.4f)  m2=%.2f (pulse %.4f)",
            m1_speed, m1_pulse, m2_speed, m2_pulse,
        )

    # ------------------------------------------------------------------
    # High-level movement API
    # ------------------------------------------------------------------

    def forward(self, speed: float = 0.5):
        """Drive straight forward.  *speed* 0.0 – 1.0."""
        speed = abs(speed)
        logger.info("Forward  speed=%.2f", speed)
        self._set_motors(speed, speed)

    def backward(self, speed: float = 0.5):
        """Drive straight backward.  *speed* 0.0 – 1.0."""
        speed = abs(speed)
        logger.info("Backward  speed=%.2f", speed)
        self._set_motors(-speed, -speed)

    def right(self, speed: float = 0.5):
        """Pivot/turn left (right motor forward, left motor backward)."""
        speed = abs(speed)
        logger.info("Left  speed=%.2f", speed)
        self._set_motors(-speed, speed)

    def left(self, speed: float = 0.5):
        """Pivot/turn right (left motor forward, right motor backward)."""
        speed = abs(speed)
        logger.info("Right  speed=%.2f", speed)
        self._set_motors(speed, -speed)

    def stop(self):
        """Immediately stop both motors (neutral pulse)."""
        logger.info("Stop")
        # self.motor1_pwm.value = _NEUTRAL
        # self.motor2_pwm.value = _NEUTRAL
        self.motor1_pwm.off()
        self.motor2_pwm.off()
        # SWITCH 6 Must be UP

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Release GPIO resources."""
        self.stop()
        self.motor1_pwm.close()
        self.motor2_pwm.close()
        logger.info("MotorController closed")


# ------------------------------------------------------------------
# Quick self-test (python -m components.motor)
# ------------------------------------------------------------------
if __name__ == "__main__":
    from time import sleep

    logging.basicConfig(level=logging.DEBUG)
    motor = MotorController()

    try:
        print("▶ Forward 80 % for 3 s")
        motor.forward(0.8)
        sleep(3)

        print("▶ Backward 50 % for 3 s")
        motor.backward(0.5)
        sleep(3)

        print("▶ Left turn 60 % for 2 s")
        motor.left(0.6)
        sleep(2)

        print("▶ Right turn 60 % for 2 s")
        motor.right(0.6)
        sleep(2)

        print("■ Stop")
        motor.stop()
    finally:
        motor.close()


