function z_all = simulate_imu_readings(t, q_true, omega_true, alpha_true, ...
                                       r_head, r_mid, r_butt, ...
                                       sigma_accel, sigma_gyro, b_gyro_true)
% SIMULATE_IMU_READINGS  Simulate noisy IMU readings for 3 rigidly-mounted sensors.
%
%   Models the accelerometer and gyroscope outputs for sensors placed at
%   different lever-arm positions on a rigid body undergoing known rotation.
%
%   Physics Model:
%   ──────────────
%   For each IMU at lever-arm position r_i (in body frame, from COM):
%
%       Accelerometer:
%           a_i = Rᵀ · g_world  +  α × r_i  +  ω × (ω × r_i)  +  noise
%                 ↑               ↑              ↑
%                 gravity in      tangential     centripetal
%                 body frame      acceleration   acceleration
%
%       Gyroscope:
%           g_i = ω  +  b_gyro  +  noise
%                 ↑      ↑
%                 true   constant
%                 rate   bias
%
%   Inputs:
%       t            — (N×1)  time vector [s]
%       q_true       — (N×4)  quaternions [q0 q1 q2 q3], scalar-first
%       omega_true   — (N×3)  angular velocity in body frame [rad/s]
%       alpha_true   — (N×3)  angular acceleration in body frame [rad/s²]
%       r_head       — (1×3)  lever arm of head IMU [m]
%       r_mid        — (1×3)  lever arm of mid IMU [m]
%       r_butt       — (1×3)  lever arm of butt IMU [m]
%       sigma_accel  — scalar, accelerometer noise std [m/s²]
%       sigma_gyro   — scalar, gyroscope noise std [rad/s]
%       b_gyro_true  — (1×3)  true constant gyroscope bias [rad/s]
%
%   Output:
%       z_all  — (N×18) stacked IMU measurements:
%                [ax1 ay1 az1 gx1 gy1 gz1 | ax2 ay2 az2 gx2 gy2 gz2 | ax3 ay3 az3 gx3 gy3 gz3]
%                 ← head (sensor 1) →       ← mid (sensor 2) →         ← butt (sensor 3) →
%
%   University Project — Multi-IMU UKF Sensor Fusion
%   ---------------------------------------------------------------

    %% ================= Constants =============================
    g_world = [0; 0; 9.81];   % Gravity vector in world frame (Z-up)
    N = length(t);

    % Pack lever arms into a cell for loop convenience
    lever_arms = {r_head(:)', r_mid(:)', r_butt(:)'};
    n_sensors  = 3;

    %% ================= Preallocate Output ====================
    z_all = zeros(N, 6 * n_sensors);   % 6 DOF per sensor × 3 sensors

    %% ================= Generate Readings =====================
    for s = 1:n_sensors
        r_i = lever_arms{s}(:);   % (3×1) lever arm vector

        % Column indices for this sensor in the output matrix
        col_accel = (s-1)*6 + (1:3);   % e.g. sensor 1 → cols 1:3
        col_gyro  = (s-1)*6 + (4:6);   % e.g. sensor 1 → cols 4:6

        for k = 1:N
            %% --- Rotation matrix: world → body ---
            % quat2rotm expects scalar-first [q0 q1 q2 q3] (1×4)
            % It returns R such that v_world = R * v_body,
            % so R' (transpose) maps world → body.
            R = quat2rotm(q_true(k,:));   % (3×3)

            %% --- Accelerometer model ---
            w = omega_true(k,:)';   % (3×1) angular velocity
            a = alpha_true(k,:)';   % (3×1) angular acceleration

            % Gravity component in body frame
            a_grav = R' * g_world;

            % Tangential acceleration:  α × r
            a_tang = cross(a, r_i);

            % Centripetal acceleration: ω × (ω × r)
            a_cent = cross(w, cross(w, r_i));

            % Total ideal accelerometer reading + noise
            a_meas = a_grav + a_tang + a_cent + sigma_accel * randn(3,1);

            %% --- Gyroscope model ---
            g_meas = w + b_gyro_true(:) + sigma_gyro * randn(3,1);

            %% --- Store ---
            z_all(k, col_accel) = a_meas';
            z_all(k, col_gyro)  = g_meas';
        end
    end

    %% ================= Summary ===============================
    fprintf('simulate_imu_readings: %d samples × %d sensors\n', N, n_sensors);
    fprintf('  Noise: σ_accel = %.4f m/s²,  σ_gyro = %.5f rad/s\n', ...
        sigma_accel, sigma_gyro);
    fprintf('  Gyro bias: [%.4f, %.4f, %.4f] rad/s\n', b_gyro_true);
    fprintf('  Lever arms (m):\n');
    fprintf('    Head: [%.3f, %.3f, %.3f]\n', lever_arms{1});
    fprintf('    Mid:  [%.3f, %.3f, %.3f]\n', lever_arms{2});
    fprintf('    Butt: [%.3f, %.3f, %.3f]\n', lever_arms{3});
end
