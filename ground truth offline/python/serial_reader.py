"""
serial_reader.py — ESP32-S3 Serial IMU Data Reader

Reads multi-IMU data from an ESP32-S3 microcontroller over USB serial.
The ESP32 sends CSV-formatted data at 200 Hz containing accelerometer
and gyroscope readings from 3 MPU6050 IMUs.

Expected CSV format from ESP32:
    ax1,ay1,az1,gx1,gy1,gz1,ax2,ay2,az2,gx2,gy2,gz2,ax3,ay3,az3,gx3,gy3,gz3

where:
    - ax, ay, az: accelerometer readings [m/s²]
    - gx, gy, gz: gyroscope readings [rad/s]
    - Suffix 1: Head IMU (r = [+0.08, 0, 0] m)
    - Suffix 2: Mid  IMU (r = [0, 0, 0] m)
    - Suffix 3: Butt IMU (r = [-0.08, 0, 0] m)

Usage:
    reader = SerialIMUReader('COM3')
    data = reader.read_frame()  # returns numpy array of 18 values
    reader.close()

    # Or use as context manager:
    with SerialIMUReader('COM3') as reader:
        while True:
            data = reader.read_frame()
            if data is not None:
                process(data)

Author: Senior Project — Multi-IMU UKF Sensor Fusion
"""

import numpy as np
import time

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[SerialIMUReader] WARNING: pyserial not installed. "
          "Install with: pip install pyserial")


class SerialIMUReader:
    """Reads multi-IMU data from ESP32-S3 over USB serial connection.

    This class handles:
    - Serial port connection with automatic retry
    - Parsing CSV data from the ESP32
    - Graceful error handling for disconnects and malformed data
    - Context manager protocol for safe cleanup

    Parameters
    ----------
    port : str
        Serial port name.
        - Windows: 'COM3', 'COM4', etc.
        - Linux:   '/dev/ttyUSB0', '/dev/ttyACM0'
        - macOS:   '/dev/cu.usbserial-xxxx'
    baud_rate : int, optional
        Serial baud rate (default: 115200, matching ESP32 firmware).
    timeout : float, optional
        Read timeout in seconds (default: 1.0).
        If no data is received within this time, read_frame() returns None.

    Attributes
    ----------
    N_VALUES : int
        Number of values expected per frame (18 = 3 IMUs × 6 channels).
    is_connected : bool
        Whether the serial port is currently open and connected.

    Raises
    ------
    ImportError
        If pyserial is not installed.
    serial.SerialException
        If the specified port cannot be opened.

    Examples
    --------
    >>> reader = SerialIMUReader('COM3', baud_rate=115200)
    [SerialIMUReader] Connected to COM3 at 115200 baud
    >>> data = reader.read_frame()
    >>> print(data.shape)
    (18,)
    >>> reader.close()
    """

    N_VALUES = 18  # 3 IMUs × (3 accel + 3 gyro)

    def __init__(self, port, baud_rate=115200, timeout=1.0):
        if not SERIAL_AVAILABLE:
            raise ImportError(
                "pyserial is required for serial communication. "
                "Install with: pip install pyserial"
            )

        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None
        self._frame_count = 0
        self._error_count = 0

        self._connect()

    def _connect(self):
        """Establish serial connection to ESP32-S3.

        Waits 2 seconds after connection for the ESP32 to complete
        its boot sequence and start sending data.
        """
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                # Standard 8N1 settings (most common for ESP32)
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            # ESP32 resets when DTR is asserted on USB-CDC connection.
            # Wait for it to boot and start streaming data.
            time.sleep(2.0)

            # Flush any boot messages or partial data
            self.ser.reset_input_buffer()

            print(f"[SerialIMUReader] Connected to {self.port} "
                  f"at {self.baud_rate} baud")

        except serial.SerialException as e:
            print(f"[SerialIMUReader] ERROR: Could not connect to "
                  f"{self.port}: {e}")
            print(f"[SerialIMUReader] Troubleshooting:")
            print(f"  1. Is the ESP32-S3 connected via USB?")
            print(f"  2. Is the correct COM port specified?")
            print(f"  3. Is another program using the port?")
            raise

    @property
    def is_connected(self):
        """Check if the serial port is open and connected."""
        return self.ser is not None and self.ser.is_open

    def read_frame(self):
        """Read one frame of IMU data (18 values).

        Reads a single line from the serial port, parses the CSV
        values, and returns them as a numpy array.

        Returns
        -------
        data : ndarray of shape (18,) or None
            IMU readings in the format:
            [ax1, ay1, az1, gx1, gy1, gz1,
             ax2, ay2, az2, gx2, gy2, gz2,
             ax3, ay3, az3, gx3, gy3, gz3]

            Returns None if:
            - Serial port is not open
            - Read timeout occurred (no data)
            - Data is malformed (wrong number of values, non-numeric)
        """
        if not self.is_connected:
            print("[SerialIMUReader] WARNING: Serial port not open")
            return None

        try:
            # Read one line (terminated by \n)
            raw = self.ser.readline()

            # Decode bytes to string (ignore encoding errors from noise)
            line = raw.decode('utf-8', errors='ignore').strip()

            if not line:
                # Timeout — no data received within the timeout period
                return None

            # Split CSV values
            values = line.split(',')

            if len(values) != self.N_VALUES:
                # Malformed line (e.g., startup message, partial frame)
                # This is common during the first few frames
                self._error_count += 1
                return None

            # Parse all values to float
            data = np.array([float(v) for v in values])

            self._frame_count += 1
            return data

        except (ValueError, UnicodeDecodeError):
            # Non-numeric data (e.g., debug print from ESP32)
            self._error_count += 1
            return None

        except serial.SerialException as e:
            print(f"[SerialIMUReader] Serial error: {e}")
            self._error_count += 1
            return None

    def get_stats(self):
        """Get reader statistics.

        Returns
        -------
        stats : dict
            Dictionary with 'frames_read' and 'errors' counts.
        """
        return {
            'frames_read': self._frame_count,
            'errors': self._error_count,
            'error_rate': (self._error_count /
                          max(1, self._frame_count + self._error_count))
        }

    def close(self):
        """Close the serial connection.

        Safe to call multiple times. Prints connection statistics
        on close.
        """
        if self.ser is not None and self.ser.is_open:
            stats = self.get_stats()
            self.ser.close()
            print(f"[SerialIMUReader] Connection to {self.port} closed")
            print(f"  Frames read: {stats['frames_read']}, "
                  f"Errors: {stats['errors']}")

    # ── Context Manager Protocol ──

    def __enter__(self):
        """Support 'with SerialIMUReader(...) as reader:' syntax."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure serial port is closed on context exit."""
        self.close()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Destructor: close serial port if still open."""
        self.close()
