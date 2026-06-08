%% RUN_SYNTHETIC_DATA  Main script for multi-IMU synthetic data generation.
%
%   This script:
%     1. Generates a smooth, known 3D rotation trajectory
%     2. Simulates noisy IMU readings for 3 sensors at lever-arm positions
%     3. Saves ground truth and sensor data to CSV files
%     4. Produces diagnostic plots
%
%   University Project — Multi-IMU UKF Sensor Fusion
%   ---------------------------------------------------------------

clear; clc; close all;
fprintf('========================================\n');
fprintf(' Multi-IMU Synthetic Data Generator\n');
fprintf('========================================\n\n');

%% =================== Physical Parameters =====================
% These MUST match the Python UKF implementation exactly.

% Lever arm positions from center of mass [m]
r_head = [0.08,  0,  0];   % +x direction
r_mid  = [0,     0,  0];   % at COM
r_butt = [-0.08, 0,  0];   % −x direction

% MPU6050 noise characteristics
sigma_gyro  = 0.0012;                    % Gyroscope noise std [rad/s]
sigma_accel = 0.056;                     % Accelerometer noise std [m/s²]
b_gyro_true = [0.02, -0.015, 0.01];     % True constant gyro bias [rad/s]

%% =================== Step 1: Generate Trajectory =============
fprintf('--- Step 1: Generating trajectory ---\n');
[t, q_true, omega_true, alpha_true, euler_true] = generate_trajectory();
fprintf('\n');

%% =================== Step 2: Simulate IMU Readings ===========
fprintf('--- Step 2: Simulating IMU readings ---\n');
z_all = simulate_imu_readings(t, q_true, omega_true, alpha_true, ...
                               r_head, r_mid, r_butt, ...
                               sigma_accel, sigma_gyro, b_gyro_true);
fprintf('\n');

%% =================== Step 3: Save to CSV =====================
fprintf('--- Step 3: Saving CSV files ---\n');

% Output directory (same as this script's directory)
out_dir = fileparts(mfilename('fullpath'));
if isempty(out_dir)
    out_dir = pwd;
end

% ----- Ground Truth CSV -----
% Columns: time, q0, q1, q2, q3, wx, wy, wz, alpha_x, alpha_y, alpha_z
gt_header = {'time','q0','q1','q2','q3','wx','wy','wz','alpha_x','alpha_y','alpha_z'};
gt_data   = [t, q_true, omega_true, alpha_true];
gt_table  = array2table(gt_data, 'VariableNames', gt_header);

gt_path = fullfile(out_dir, 'ground_truth.csv');
writetable(gt_table, gt_path);
fprintf('  Saved: %s  (%d rows × %d cols)\n', gt_path, size(gt_data,1), size(gt_data,2));

% ----- Synthetic IMU CSV -----
% Columns: time, ax1..gz1, ax2..gz2, ax3..gz3  (18 sensor columns)
imu_header = {'time', ...
    'ax1','ay1','az1','gx1','gy1','gz1', ...
    'ax2','ay2','az2','gx2','gy2','gz2', ...
    'ax3','ay3','az3','gx3','gy3','gz3'};
imu_data  = [t, z_all];
imu_table = array2table(imu_data, 'VariableNames', imu_header);

imu_path = fullfile(out_dir, 'synthetic_imu.csv');
writetable(imu_table, imu_path);
fprintf('  Saved: %s  (%d rows × %d cols)\n', imu_path, size(imu_data,1), size(imu_data,2));

%% =================== Step 4: Diagnostic Plots ================
fprintf('\n--- Step 4: Generating plots ---\n');

figure('Name', 'Multi-IMU Synthetic Data', ...
       'NumberTitle', 'off', ...
       'Position', [100, 100, 1200, 900], ...
       'Color', 'w');

% ---- Subplot 1: Euler Angles ----
subplot(2,2,1);
plot(t, rad2deg(euler_true(:,3)), 'b-', 'LineWidth', 1.2); hold on;
plot(t, rad2deg(euler_true(:,2)), 'r-', 'LineWidth', 1.2);
plot(t, rad2deg(euler_true(:,1)), 'g-', 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Angle [°]');
title('Ground Truth Euler Angles');
legend('Yaw ψ', 'Pitch θ', 'Roll φ', 'Location', 'best');
grid on; set(gca, 'FontSize', 10);

% ---- Subplot 2: Angular Velocity ----
subplot(2,2,2);
plot(t, omega_true(:,1), 'b-', 'LineWidth', 1.0); hold on;
plot(t, omega_true(:,2), 'r-', 'LineWidth', 1.0);
plot(t, omega_true(:,3), 'g-', 'LineWidth', 1.0);
xlabel('Time [s]'); ylabel('ω [rad/s]');
title('Ground Truth Angular Velocity (Body Frame)');
legend('ω_x', 'ω_y', 'ω_z', 'Location', 'best');
grid on; set(gca, 'FontSize', 10);

% ---- Subplot 3: Accelerometer — Sensor 1 vs Sensor 3 ----
% Shows the lever-arm effect: sensors at opposite ends see different accel.
subplot(2,2,3);
% Sensor 1 (head): columns 1:3
% Sensor 3 (butt): columns 13:15
plot(t, z_all(:,1), 'b-', 'LineWidth', 0.6); hold on;
plot(t, z_all(:,13), 'b--', 'LineWidth', 0.6);
plot(t, z_all(:,2), 'r-', 'LineWidth', 0.6);
plot(t, z_all(:,14), 'r--', 'LineWidth', 0.6);
plot(t, z_all(:,3), 'g-', 'LineWidth', 0.6);
plot(t, z_all(:,15), 'g--', 'LineWidth', 0.6);
xlabel('Time [s]'); ylabel('Accel [m/s²]');
title('Accelerometer: Head (solid) vs Butt (dashed)');
legend('Head a_x','Butt a_x','Head a_y','Butt a_y','Head a_z','Butt a_z', ...
       'Location', 'best', 'FontSize', 7);
grid on; set(gca, 'FontSize', 10);

% ---- Subplot 4: Gyroscope — All 3 Sensors ----
% Should look similar (same ω), differing only by noise realization.
subplot(2,2,4);
% Sensor 1 gyro: cols 4:6, Sensor 2: cols 10:12, Sensor 3: cols 16:18
plot(t, z_all(:,4), 'b-',  'LineWidth', 0.6); hold on;
plot(t, z_all(:,10), 'b--', 'LineWidth', 0.6);
plot(t, z_all(:,16), 'b:',  'LineWidth', 0.8);
plot(t, z_all(:,5), 'r-',  'LineWidth', 0.6);
plot(t, z_all(:,11), 'r--', 'LineWidth', 0.6);
plot(t, z_all(:,17), 'r:',  'LineWidth', 0.8);
plot(t, z_all(:,6), 'g-',  'LineWidth', 0.6);
plot(t, z_all(:,12), 'g--', 'LineWidth', 0.6);
plot(t, z_all(:,18), 'g:',  'LineWidth', 0.8);
xlabel('Time [s]'); ylabel('Gyro [rad/s]');
title('Gyroscope: Head (solid), Mid (dashed), Butt (dotted)');
legend('H g_x','M g_x','B g_x', 'H g_y','M g_y','B g_y', 'H g_z','M g_z','B g_z', ...
       'Location', 'best', 'FontSize', 6);
grid on; set(gca, 'FontSize', 10);

% Add a super-title
sgtitle('Multi-IMU Synthetic Data — UKF Sensor Fusion Project', ...
        'FontSize', 14, 'FontWeight', 'bold');

% Save figure as PNG
fig_path = fullfile(out_dir, 'synthetic_data_plots.png');
exportgraphics(gcf, fig_path, 'Resolution', 150);
fprintf('  Saved figure: %s\n', fig_path);

fprintf('\n========================================\n');
fprintf(' Done! Files saved to: %s\n', out_dir);
fprintf('========================================\n');
