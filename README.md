# gyroMouse

gyroMouse is a comprehensive project that transforms a 3-sensor IMU device into an "air mouse" using an Unscented Kalman Filter (UKF-lite). It uses live accelerometer and gyroscope data to accurately estimate angular velocity ($\omega$) and seamlessly translates those movements into PC cursor coordinates.

The repository is divided into two main components:
1. **Live MATLAB Application**: Real-time sensor processing and mouse control using Java's `AWT Robot`.
2. **Ground Truth & Offline Engine**: Both MATLAB and Python implementations for dataset logging, synthetic data generation, and offline algorithm validation.

---

## 📑 Features

- **Live UKF-lite Algorithm**: Specifically tuned for fast-changing angular velocities and slowly drifting gyroscope biases. Includes stationary detection for zero-velocity updates to prevent cursor drift.
- **Real-Time Serial Communication**: Captures 18-element arrays (accelerometer and gyroscope data from center, left, and right sensors).
- **Direct PC Mouse Control**: Maps estimated angular velocities (pitch/yaw) directly into screen X/Y movements with adjustable deadzones and sensitivity gains.
- **Automatic Calibration**: Computes initial biases and noise covariances by holding the device stationary upon launch.
- **Offline Analytics**: A dedicated Python and MATLAB suite to simulate IMU readings, process logged datasets, and fine-tune Kalman filter parameters without needing live hardware.

---

## 🛠️ Prerequisites & Setup

### Hardware Requirements
- A custom IMU setup continuously sending comma-separated data over a Serial/USB connection (18 values per line: `acc_center(3), gyro_center(3), acc_left(3), gyro_left(3), acc_right(3), gyro_right(3)`).

### Software Requirements

#### 1. MATLAB (For Live Mouse Control & Offline Analytics)
- MATLAB (tested with recent versions).
- Must support `serialport` and `tiledlayout` (R2019b or newer recommended).
- Java Runtime Environment (usually bundled with MATLAB) for `java.awt.Robot` to control the mouse.

#### 2. Python (For Ground Truth & Simulation)
- Python 3.8+
- Create a virtual environment and install the required dependencies:
  ```bash
  cd "ground truth offline/python"
  python -m venv venv
  
  # Windows
  .\venv\Scripts\activate
  # Mac/Linux
  source venv/bin/activate
  
  pip install -r requirements.txt
  ```
  *(Dependencies include `numpy`, `scipy`, `matplotlib`, and `pyserial`).*

---

## 🚀 How to Run

### Part A: Live Mouse Control (MATLAB)

1. **Connect Hardware**: Plug your IMU device into the computer. Note the COM port it is assigned (e.g., `COM7`).
2. **Configure Port**: 
   - Open `live_ukf_lite_serial.m` in MATLAB.
   - Locate the variables at the top of the file and update `portAdi` and `baudRate` to match your device settings:
     ```matlab
     portAdi = "COM7";     % Change to your specific port
     baudRate = 115200;    % Change if your device uses a different baud rate
     ```
3. **Execute**: 
   - Open and run the `live_ukf_lite_mouse.m` script. This script sets `enable_mouse = true` and launches the system.
4. **Calibration Process**:
   - **CRITICAL**: The moment you run the script, keep the IMU device **perfectly still on a flat surface** for exactly 5 seconds.
   - The system will calculate the static gyro biases. You will see a "Kalibrasyon tamamlandi" (Calibration completed) message in the MATLAB command window once it's done.
5. **Usage**:
   - Pick up the IMU device and move it around. The mouse cursor will move relative to the angular velocity (gyroscope readings).
   - If the cursor drifts, place the device still on the table for a second. The `is_still` detection will automatically dampen the values and adapt to the new bias.
   - To stop the program, press `Ctrl+C` in the MATLAB Command Window.

### Part B: Offline Processing & Synthetic Data

If you want to debug the UKF algorithm, tune parameters, or run datasets without holding the physical device:

**Using MATLAB:**
- Run `log_ukf_dataset.m` to log real serial data to a `.csv` file.
- Run `offline_ukf_dataset.m` to play back that dataset through the UKF algorithm and visualize the NIS (Normalized Innovation Squared) and state estimates.
- Check out the `ground truth offline/matlab/` folder for generating synthetic trajectories to perfectly validate the filter's math.

**Using Python:**
- Ensure your `venv` is activated.
- You can run the offline UKF filter written entirely in Python by executing:
  ```bash
  cd "ground truth offline/python"
  python run_offline.py
  ```
- Or run the live serial monitor in Python:
  ```bash
  python run_live.py
  ```

---

## ⚙️ Configuration & Tuning

In `live_ukf_lite_serial.m`, you can adjust the air-mouse feel by modifying the following variables:
- `mouse_deadzone = 0.035;` : Minimum angular velocity required to move the mouse (prevents jitter).
- `mouse_gain_x = 18;` : Sensitivity multiplier for the X-axis.
- `mouse_gain_y = -18;` : Sensitivity multiplier for the Y-axis.
- `max_mouse_step = 35;` : Maximum pixels the cursor can jump per frame (caps maximum speed).

---

## 📝 License
This project is open-source. Feel free to modify the Unscented Kalman Filter parameters or port the mouse controlling interface to other languages.
