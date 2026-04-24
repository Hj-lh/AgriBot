"""
Motor Controller Component
==========================
Differential-drive motor controller using Simplified Serial Mode.
Communicates via hardware UART (GPIO 14 / /dev/serial0) at 9600 baud.
"""

import logging
import serial

logger = logging.getLogger(__name__)

class MotorController:
    """High-level differential-drive controller via Serial."""

    def __init__(self, port='/dev/serial0', baudrate=9600):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            logger.info("MotorController initialized via Serial (%s at %d baud)", port, baudrate)
        except serial.SerialException as e:
            logger.error("Failed to open serial port: %s", e)
            raise

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _set_motors(self, m1_speed: float, m2_speed: float):
        """
        Set individual motor speeds.
        m1_speed, m2_speed : -1.0 (reverse) to 1.0 (forward)
        """
        m1_speed = max(-1.0, min(1.0, m1_speed))
        m2_speed = max(-1.0, min(1.0, m2_speed))

        # Motor 1: 1 (reverse) to 127 (forward), 64 is center/stop
        m1_cmd = int(64 + (m1_speed * 63))
        
        # Motor 2: 128 (reverse) to 255 (forward), 192 is center/stop
        # Negate m2_speed because the physical motor is mounted in reverse
        m2_cmd = int(192 + (-m2_speed * 63))

        # Write the two bytes directly to the Sabertooth
        self.ser.write(bytes([m1_cmd, m2_cmd]))

        logger.debug("Motors set m1=%.2f (Cmd: %d) m2=%.2f (Cmd: %d)", 
                     m1_speed, m1_cmd, m2_speed, m2_cmd)

    # ------------------------------------------------------------------
    # High-level movement API
    # ------------------------------------------------------------------

    def forward(self, speed: float = 0.5):
        speed = abs(speed)
        logger.info("Forward speed=%.2f", speed)
        self._set_motors(speed, speed)

    def backward(self, speed: float = 0.5):
        speed = abs(speed)
        logger.info("Backward speed=%.2f", speed)
        self._set_motors(-speed, -speed)

    def right(self, speed: float = 0.5):
        speed = abs(speed)
        logger.info("Right speed=%.2f", speed)
        self._set_motors(speed, -speed)

    def left(self, speed: float = 0.5):
        speed = abs(speed)
        logger.info("Left speed=%.2f", speed)
        self._set_motors(-speed, speed)

    def stop(self):
        """Send byte 0 to immediately shut down both motors."""
        logger.info("Stop")
        self.ser.write(bytes([0]))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        self.stop()
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        logger.info("MotorController closed")


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