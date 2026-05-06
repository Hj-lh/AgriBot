import logging
import serial
import time

# Create the logger for this module
logger = logging.getLogger(__name__)

class MotorController:
    """High-level differential-drive controller via Sabertooth Packetized Serial."""

    def __init__(self, port='/dev/ttyAMA0', baudrate=9600, address=128):
        self.address = address
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            
            # Packetized Serial requires a 2-second startup delay 
            # and a bauding character (170) to sync the frequency.
            logger.info("Waiting 2 seconds for Sabertooth startup...")
            time.sleep(2) 
            
            logger.info("Sending bauding character (170)...")
            self.ser.write(bytearray([170])) 
            
            logger.info("MotorController initialized (Address: %d)", address)
        except serial.SerialException as e:
            logger.error("Failed to open serial port: %s", e)
            raise

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _send_packet(self, command, data):
        """Constructs and sends a 4-byte packet: [Addr, Cmd, Data, Checksum]"""
        # Checksum = (Address + Command + Data) & 127
        checksum = (self.address + command + data) & 0b01111111
        packet = bytearray([self.address, command, data, checksum])
        self.ser.write(packet)

    def _set_motors(self, m1_speed: float, m2_speed: float):
        """
        Set individual motor speeds.
        m1_speed, m2_speed : -1.0 (reverse) to 1.0 (forward)
        """
        # Motor 1: Command 0 is Forward, Command 1 is Reverse
        m1_cmd = 0 if m1_speed >= 0 else 1
        m1_data = int(abs(m1_speed) * 127)
        m1_data = max(0, min(127, m1_data)) 

        # Motor 2: Command 4 is Forward, Command 5 is Reverse
        m2_cmd = 4 if m2_speed >= 0 else 5
        m2_data = int(abs(m2_speed) * 127)
        m2_data = max(0, min(127, m2_data))

        self._send_packet(m1_cmd, m1_data)
        self._send_packet(m2_cmd, m2_data)

        logger.debug("M1 Speed: %.2f (Cmd: %d, Data: %d) | M2 Speed: %.2f (Cmd: %d, Data: %d)", 
                     m1_speed, m1_cmd, m1_data, m2_speed, m2_cmd, m2_data)

    # ------------------------------------------------------------------
    # High-level movement API
    # ------------------------------------------------------------------

    def forward(self, speed: float = 0.5):
        logger.info("Moving Forward at %.2f", speed)
        self._set_motors(speed, speed)

    def backward(self, speed: float = 0.5):
        logger.info("Moving Backward at %.2f", speed)
        self._set_motors(-speed, -speed)

    def right(self, speed: float = 0.5):
        logger.info("Turning Right at %.2f", speed)
        self._set_motors(speed, -speed)

    def left(self, speed: float = 0.5):
        logger.info("Turning Left at %.2f", speed)
        self._set_motors(-speed, speed)

    def stop(self):
        """Send speed 0 to both motors to stop."""
        self._send_packet(0, 0)
        self._send_packet(4, 0)
        logger.info("Motors Stopped")

    def close(self):
        self.stop()
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        logger.info("MotorController Serial Port Closed")

# ------------------------------------------------------------------
# Main test block
# ------------------------------------------------------------------
if __name__ == "__main__":
    # This part configures the logger to print to your terminal screen.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    motor = MotorController()
    try:
        motor.forward(0.4)
        time.sleep(2)
        motor.stop()
    finally:
        motor.close()
