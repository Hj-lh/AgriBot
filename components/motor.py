"""
Motor Controller Component
==========================
Differential-drive motor controller using dual BTS7960 drivers.
Communicates via hardware PWM using the gpiozero library.
"""

import logging
from gpiozero import Robot, DigitalOutputDevice

logger = logging.getLogger(__name__)

class MotorController:
    """High-level differential-drive controller via PWM."""

    def __init__(self, 
                 left_fwd_pin=12, left_bwd_pin=13, left_en_pin=5,
                 right_fwd_pin=22, right_bwd_pin=23, right_en_pin=6,
                 pwm_frequency=1000):
        
        # --------------------------------------------------------------
        # SOFTWARE TRIM (Hardcoded here to keep main.py clean)
        # Adjust these values (0.0 to 1.0) to make the robot drive straight
        # --------------------------------------------------------------
        self.left_trim = 1.0
        self.right_trim = 0.9 
        
        try:
            # 1. SETUP ENABLE PINS 
            # (Note: If you wired EN pins directly to 3.3V, you can ignore/delete this block)
            self.left_en = DigitalOutputDevice(left_en_pin)
            self.right_en = DigitalOutputDevice(right_en_pin)
            
            # Turn the motor drivers "ON" (Awake)
            self.left_en.on()
            self.right_en.on()

            # 2. SETUP PWM PINS
            self.robot = Robot(left=(left_fwd_pin, left_bwd_pin), 
                               right=(right_fwd_pin, right_bwd_pin))
            
            # 3. OVERRIDE FREQUENCY (Prevents motor whining)
            self.robot.left_motor.forward_device.frequency = pwm_frequency
            self.robot.left_motor.backward_device.frequency = pwm_frequency
            self.robot.right_motor.forward_device.frequency = pwm_frequency
            self.robot.right_motor.backward_device.frequency = pwm_frequency
            
            logger.info("MotorController initialized with Enables and Trim.")
            
        except Exception as e:
            logger.error("Failed to initialize BTS7960 PWM pins: %s", e)
            raise

    # ------------------------------------------------------------------
    # Low-level helper
    # ------------------------------------------------------------------

    def _set_robot_value(self, left_speed: float, right_speed: float):
        """Applies trim multipliers and sets the gpiozero Robot value."""
        
        # Apply the trim multipliers
        l_cmd = left_speed * self.left_trim
        r_cmd = right_speed * self.right_trim
        
        # Ensure we don't accidentally exceed the -1.0 to 1.0 limit
        l_cmd = max(-1.0, min(1.0, l_cmd))
        r_cmd = max(-1.0, min(1.0, r_cmd))
        
        self.robot.value = (l_cmd, r_cmd)

    # ------------------------------------------------------------------
    # High-level movement API
    # ------------------------------------------------------------------

    def forward(self, speed: float = 0.5):
        speed = abs(speed)
        logger.info("Forward speed=%.2f", speed)
        self._set_robot_value(speed, speed)

    def backward(self, speed: float = 0.5):
        speed = abs(speed)
        logger.info("Backward speed=%.2f", speed)
        self._set_robot_value(-speed, -speed)

    def right(self, speed: float = 0.5):
        speed = abs(speed)
        logger.info("Right speed=%.2f", speed)
        self._set_robot_value(speed, -speed)

    def left(self, speed: float = 0.5):
        speed = abs(speed)
        logger.info("Left speed=%.2f", speed)
        self._set_robot_value(-speed, speed)

    def stop(self):
        """Immediately shut down both motors."""
        logger.info("Stop")
        self.robot.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        self.stop()
        
        # Put the motor drivers to sleep safely
        if hasattr(self, 'left_en'):
            self.left_en.off()
            self.right_en.off()
            
        if hasattr(self, 'robot'):
            self.robot.close()
            
        logger.info("MotorController closed")


if __name__ == "__main__":
    from time import sleep

    logging.basicConfig(level=logging.DEBUG)
    motor = MotorController()

    try:
        print("▶ Forward 80% for 3 s")
        motor.forward(0.8)
        sleep(3)

        print("▶ Backward 50% for 3 s")
        motor.backward(0.5)
        sleep(3)

        print("■ Stop")
        motor.stop()
    finally:
        motor.close()