"""
run_live.py — Real-Time UKF with Live ESP32-S3 IMU Data

Reads multi-IMU data from an ESP32-S3 over USB serial and runs the
Unscented Kalman Filter in real-time for orientation estimation.

The script displays estimated Euler angles and angular velocities in
the console, with an optional live matplotlib plot.

Usage:
    python run_live.py COM3
    python run_live.py --port COM3 --baud 115200
    python run_live.py --port COM3 --plot

Controls:
    Ctrl+C : Gracefully stop and print summary

Author: Senior Project — Multi-IMU UKF Sensor Fusion
"""

import argparse
import os
import sys
import time
import numpy as np

# Add the current directory to the path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quaternion_utils import quat_to_euler
from models import process_model, measurement_model, SIGMA_ACCEL, SIGMA_GYRO
from ukf import UKF
from serial_reader import SerialIMUReader


# ────────────────────────────────────────────────────────────────────
# UKF Initialization (same parameters as offline version)
# ────────────────────────────────────────────────────────────────────

def initialize_ukf():
    """Initialize the UKF with standard parameters.

    Uses the same tuning as the offline validation to ensure
    consistent behavior between offline and live modes.

    Returns
    -------
    ukf_filter : UKF
        Fully initialized UKF instance ready for processing.
    """
    n_state = 13  # [q0,q1,q2,q3, wx,wy,wz, ax,ay,az, bx,by,bz]
    n_meas = 18   # 3 IMUs × (3 accel + 3 gyro)

    # ── Initial state: at rest, upright ──
    x0 = np.zeros(n_state)
    x0[0] = 1.0  # identity quaternion

    # ── Initial covariance ──
    P0 = np.diag([
        1e-4, 1e-4, 1e-4, 1e-4,    # quaternion: fairly certain
        0.1, 0.1, 0.1,              # angular velocity
        0.1, 0.1, 0.1,              # angular acceleration
        0.1, 0.1, 0.1               # gyro bias
    ])

    # ── Process noise ──
    Q = np.diag([
        1e-6, 1e-6, 1e-6, 1e-6,    # quaternion
        1e-4, 1e-4, 1e-4,           # angular velocity
        1e-2, 1e-2, 1e-2,           # angular acceleration
        1e-8, 1e-8, 1e-8            # gyro bias
    ])

    # ── Measurement noise (from MPU6050 specs) ──
    R_diag = np.zeros(n_meas)
    for i in range(3):  # 3 IMUs
        R_diag[i*6 : i*6 + 3] = SIGMA_ACCEL**2    # accelerometer
        R_diag[i*6 + 3 : i*6 + 6] = SIGMA_GYRO**2  # gyroscope
    R = np.diag(R_diag)

    # ── Create and initialize UKF ──
    ukf_filter = UKF(
        n_state, n_meas,
        process_model, measurement_model,
        Q, R,
        alpha=1e-3, beta=2, kappa=0
    )
    ukf_filter.x = x0.copy()
    ukf_filter.P = P0.copy()

    return ukf_filter


# ────────────────────────────────────────────────────────────────────
# Live Plot Setup
# ────────────────────────────────────────────────────────────────────

def setup_live_plot():
    """Create an interactive matplotlib figure for real-time display.

    Returns
    -------
    fig : Figure
    axes : array of Axes
    lines_euler : list of Line2D (3 lines for roll, pitch, yaw)
    lines_omega : list of Line2D (3 lines for wx, wy, wz)
    plot_data : dict with storage arrays and index
    """
    import matplotlib
    matplotlib.use('TkAgg')  # Use an interactive backend
    import matplotlib.pyplot as plt

    plt.ion()  # Enable interactive mode

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle('Real-Time UKF Orientation Estimation',
                 fontsize=13, fontweight='bold')

    # Rolling buffer size (show last N points at ~40Hz update rate)
    max_points = 500

    # Color palette
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    # ── Euler angle subplot ──
    lines_euler = []
    euler_labels = ['Roll', 'Pitch', 'Yaw']
    for label, color in zip(euler_labels, colors):
        line, = axes[0].plot([], [], color=color, label=label, linewidth=1.5)
        lines_euler.append(line)
    axes[0].set_ylabel('Angle (°)')
    axes[0].set_xlabel('Time (s)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_title('Euler Angles')
    axes[0].grid(True, alpha=0.3)

    # ── Angular velocity subplot ──
    lines_omega = []
    omega_labels = ['ωx', 'ωy', 'ωz']
    for label, color in zip(omega_labels, colors):
        line, = axes[1].plot([], [], color=color, label=label, linewidth=1.5)
        lines_omega.append(line)
    axes[1].set_ylabel('Angular Velocity (rad/s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_title('Angular Velocity')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)

    # Storage for rolling data
    plot_data = {
        'euler': np.zeros((max_points, 3)),
        'omega': np.zeros((max_points, 3)),
        'time': np.zeros(max_points),
        'idx': 0,
        'max_points': max_points
    }

    return fig, axes, lines_euler, lines_omega, plot_data


def update_live_plot(fig, axes, lines_euler, lines_omega, plot_data,
                     euler_deg, omega, current_time):
    """Update the live plot with new data.

    Uses a circular buffer to show a rolling window of recent data.

    Parameters
    ----------
    fig : Figure
    axes : array of Axes
    lines_euler, lines_omega : list of Line2D
    plot_data : dict with circular buffer arrays
    euler_deg : ndarray, shape (3,) — [roll, pitch, yaw] in degrees
    omega : ndarray, shape (3,) — [wx, wy, wz] in rad/s
    current_time : float — elapsed time in seconds
    """
    import matplotlib.pyplot as plt

    idx = plot_data['idx']
    max_pts = plot_data['max_points']
    buf_idx = idx % max_pts

    plot_data['euler'][buf_idx] = euler_deg
    plot_data['omega'][buf_idx] = omega
    plot_data['time'][buf_idx] = current_time
    plot_data['idx'] = idx + 1

    # Get ordered data from circular buffer
    n_pts = min(idx + 1, max_pts)
    if idx + 1 <= max_pts:
        t_data = plot_data['time'][:n_pts]
        e_data = plot_data['euler'][:n_pts]
        o_data = plot_data['omega'][:n_pts]
    else:
        order = np.roll(np.arange(max_pts), -(buf_idx + 1))
        t_data = plot_data['time'][order]
        e_data = plot_data['euler'][order]
        o_data = plot_data['omega'][order]

    # Update line data
    for j in range(3):
        lines_euler[j].set_data(t_data, e_data[:, j])
        lines_omega[j].set_data(t_data, o_data[:, j])

    # Rescale axes
    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    # Redraw
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ────────────────────────────────────────────────────────────────────
# Main Real-Time Loop
# ────────────────────────────────────────────────────────────────────

def run_live(port, baud_rate=115200, enable_plot=False):
    """Run UKF in real-time with live serial data from ESP32-S3.

    Parameters
    ----------
    port : str
        Serial port (e.g., 'COM3').
    baud_rate : int
        Serial baud rate (default: 115200).
    enable_plot : bool
        If True, display a live matplotlib plot of Euler angles
        and angular velocity.
    """
    dt = 0.005  # 200 Hz sampling

    # ── Initialize UKF ──
    ukf_filter = initialize_ukf()
    print("[run_live] UKF initialized successfully")
    print("[run_live] State vector: [q0,q1,q2,q3, ωx,ωy,ωz, αx,αy,αz, bx,by,bz]")
    print("=" * 80)

    # ── Set up live plot (if enabled) ──
    fig, axes, lines_euler, lines_omega, plot_data = (
        None, None, None, None, None
    )
    if enable_plot:
        try:
            fig, axes, lines_euler, lines_omega, plot_data = setup_live_plot()
            print("[run_live] Live plotting enabled")
        except Exception as e:
            print(f"[run_live] WARNING: Could not set up plotting: {e}")
            print("[run_live] Continuing without plot...")
            enable_plot = False

    # ── Connect to ESP32-S3 ──
    try:
        reader = SerialIMUReader(port, baud_rate)
    except Exception as e:
        print(f"\n[run_live] FATAL: Could not connect to {port}: {e}")
        return

    # ── Tracking variables ──
    frame_count = 0
    valid_frames = 0
    start_time = time.time()

    # ── Print header ──
    print(f"\n[run_live] Streaming from {port}... Press Ctrl+C to stop\n")
    header = (f"{'Frame':>8}  {'Time':>7}  "
              f"{'Roll':>8}  {'Pitch':>8}  {'Yaw':>8}  "
              f"{'ωx':>8}  {'ωy':>8}  {'ωz':>8}")
    print(header)
    print("-" * len(header))

    try:
        while True:
            # Read one frame from serial
            z = reader.read_frame()

            if z is None:
                # No data or malformed — skip
                continue

            valid_frames += 1

            # ── Run UKF: Predict → Update ──
            ukf_filter.predict(dt)
            ukf_filter.update(z)

            frame_count += 1
            current_time = time.time() - start_time

            # ── Extract state estimates ──
            euler = quat_to_euler(ukf_filter.x[0:4])
            euler_deg = np.degrees(euler)  # [yaw, pitch, roll]
            omega = ukf_filter.x[4:7]

            # Reorder to [roll, pitch, yaw] for display
            roll_deg  = euler_deg[2]
            pitch_deg = euler_deg[1]
            yaw_deg   = euler_deg[0]

            # ── Console output (every 20 frames ≈ 10 Hz) ──
            if frame_count % 20 == 0:
                print(f"{frame_count:>8d}  {current_time:>7.2f}  "
                      f"{roll_deg:>8.2f}  {pitch_deg:>8.2f}  {yaw_deg:>8.2f}  "
                      f"{omega[0]:>8.4f}  {omega[1]:>8.4f}  {omega[2]:>8.4f}")

            # ── Update live plot (every 5 frames ≈ 40 Hz) ──
            if enable_plot and frame_count % 5 == 0:
                euler_display = np.array([roll_deg, pitch_deg, yaw_deg])
                update_live_plot(fig, axes, lines_euler, lines_omega,
                                 plot_data, euler_display, omega,
                                 current_time)

    except KeyboardInterrupt:
        # ── Graceful shutdown ──
        elapsed = time.time() - start_time
        actual_rate = frame_count / max(elapsed, 1e-6)

        print("\n" + "=" * 60)
        print("  Session Summary")
        print("=" * 60)
        print(f"  Duration:        {elapsed:.1f} seconds")
        print(f"  Frames processed: {frame_count}")
        print(f"  Effective rate:   {actual_rate:.1f} Hz")
        print(f"  Serial stats:     {reader.get_stats()}")

        # Print final orientation estimate
        euler = quat_to_euler(ukf_filter.x[0:4])
        print(f"\n  Final Orientation (degrees):")
        print(f"    Roll:  {np.degrees(euler[2]):>8.3f}°")
        print(f"    Pitch: {np.degrees(euler[1]):>8.3f}°")
        print(f"    Yaw:   {np.degrees(euler[0]):>8.3f}°")

        # Print estimated biases
        bias = ukf_filter.x[10:13]
        print(f"\n  Estimated Gyro Bias (rad/s):")
        print(f"    bx: {bias[0]:>10.6f}")
        print(f"    by: {bias[1]:>10.6f}")
        print(f"    bz: {bias[2]:>10.6f}")
        print("=" * 60)

    finally:
        reader.close()
        if enable_plot:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.close('all')


# ────────────────────────────────────────────────────────────────────
# Command-Line Interface
# ────────────────────────────────────────────────────────────────────

def main():
    """Parse command-line arguments and start the live UKF session."""
    parser = argparse.ArgumentParser(
        description='Real-time UKF orientation estimation with ESP32-S3 IMU data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_live.py COM3                  # Basic usage
  python run_live.py --port COM3 --plot    # With live plotting
  python run_live.py --port /dev/ttyUSB0   # Linux
        """
    )

    # Positional argument (optional, for convenience)
    parser.add_argument(
        'port_pos', nargs='?', default=None,
        help='Serial port (positional argument, e.g., COM3)'
    )

    # Named arguments
    parser.add_argument(
        '--port', '-p', type=str, default=None,
        help='Serial port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)'
    )
    parser.add_argument(
        '--baud', '-b', type=int, default=115200,
        help='Serial baud rate (default: 115200)'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Enable live matplotlib plot of Euler angles and angular velocity'
    )

    args = parser.parse_args()

    # Determine the serial port (positional takes precedence)
    port = args.port_pos or args.port
    if port is None:
        parser.error(
            "Please specify a serial port.\n"
            "  Example: python run_live.py COM3\n"
            "  Example: python run_live.py --port /dev/ttyUSB0"
        )

    # Start the live session
    run_live(port, args.baud, args.plot)


if __name__ == '__main__':
    main()
