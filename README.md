# gyroMouse

gyroMouse is a MATLAB-based project that implements an "air mouse" using an Unscented Kalman Filter (UKF-lite). It uses sensor data (accelerometer and gyroscope) to estimate angular velocity and map it to PC mouse cursor movements.

## Features
- **Live UKF-lite algorithm**: Specifically tuned for fast changing angular velocity ($\omega$) and slowly changing gyro biases.
- **Serial Communication**: Reads live sensor data (accelerometer and gyroscope values from 3 different sensor points) via serial port.
- **PC Mouse Control**: Maps the estimated angular velocity into on-screen mouse movements using `java.awt.Robot`.
- **Calibration**: Performs stationary calibration to determine initial gyro biases and noise covariance.
- **Still Detection**: Zero-velocity update and drift reduction when the device is stationary.

## Prerequisites
- MATLAB (tested with recent versions supporting `serialport` and `tiledlayout`).
- Java (for `java.awt.Robot` to control the mouse pointer).
- An IMU device continuously sending data over serial.

## Getting Started

1. **Hardware Setup**: Connect your IMU device to your computer via USB/Serial.
2. **Configuration**: Open `live_ukf_lite_serial.m` and set the correct `portAdi` (e.g., `"COM7"`) and `baudRate` (e.g., `115200`).
3. **Run**: Execute `live_ukf_lite_mouse.m` in MATLAB. This script ensures mouse control is enabled and starts the main script.
4. **Calibration**: Keep the device perfectly still for the first 5 seconds while it calibrates the gyroscope biases.
5. **Usage**: After calibration, the script will map the physical movement of the device to the mouse cursor on your screen. To stop, press `Ctrl+C` in the MATLAB command window.

## Files
- `live_ukf_lite_mouse.m`: Entry point to run the live UKF-lite with mouse control enabled.
- `live_ukf_lite_serial.m`: Main script containing serial communication, UKF algorithm, and Java robot mouse control.
- `log_ukf_dataset.m` & `offline_ukf_dataset.m`: Scripts for logging dataset and running the UKF offline (presumably for testing and validation).

## Note
Ensure the serial port is not occupied by other programs (e.g., Arduino IDE Serial Monitor) before running the MATLAB script.
