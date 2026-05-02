"""
Motor Controller Component
==========================
Differential-drive motor controller using dual BTS7960 drivers.
Communicates via hardware PWM using the gpiozero library.
"""

import logging
from gpiozero import Robot

logger = logging.getLogger(__name__)

class MotorController:
    """High-level differential-drive controller via PWM."""

    def __init__(self, 
                 left_fwd_pin=12, left_bwd_pin=13, 
                 right_fwd_pin=22, right_bwd_pin=23,
                 pwm_frequency=1000):
        
        try:
            # Initialize the differential drive robot
            self.robot = Robot(left=(left_fwd_pin, left_bwd_pin), 
                               right=(right_fwd_pin, right_bwd_pin))
            
            # OVERRIDE DEFAULT FREQUENCY (100Hz) TO PREVENT MOTOR WHINE
            # Access the underlying PWMOutputDevice objects to set custom frequency
            self.robot.left_motor.forward_device.frequency = pwm_frequency
            self.robot.left_motor.backward_device.frequency = pwm_frequency
            self.robot.right_motor.forward_device.frequency = pwm_frequency
            self.robot.right_motor.backward_device.frequency = pwm_frequency
            
            logger.info("MotorController initialized for BTS7960 modules.")
            logger.info(f"Pins: L({left_fwd_pin},{left_bwd_pin}) R({right_fwd_pin},{right_bwd_pin})")
            logger.info(f"PWM Frequency set to: {pwm_frequency}Hz")
            
        except Exception as e:
            logger.error("Failed to initialize BTS7960 PWM pins: %s", e)
            raise

    # ------------------------------------------------------------------
    # High-level movement API
    # ------------------------------------------------------------------

    def forward(self, speed: float = 0.5):
        speed = max(0.0, min(1.0, abs(speed)))
        logger.info("Forward speed=%.2f", speed)
        self.robot.forward(speed)

    def backward(self, speed: float = 0.5):
        speed = max(0.0, min(1.0, abs(speed)))
        logger.info("Backward speed=%.2f", speed)
        self.robot.backward(speed)

    def right(self, speed: float = 0.5):
        speed = max(0.0, min(1.0, abs(speed)))
        logger.info("Right speed=%.2f", speed)
        self.robot.right(speed)

    def left(self, speed: float = 0.5):
        speed = max(0.0, min(1.0, abs(speed)))
        logger.info("Left speed=%.2f", speed)
        self.robot.left(speed)

    def stop(self):
        """Immediately shut down both motors."""
        logger.info("Stop")
        self.robot.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        self.stop()
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

        print("▶ Left turn 60% for 2 s")
        motor.left(0.6)
        sleep(2)

        print("▶ Right turn 60% for 2 s")
        motor.right(0.6)
        sleep(2)

        print("■ Stop")
        motor.stop()
    finally:
        motor.close()