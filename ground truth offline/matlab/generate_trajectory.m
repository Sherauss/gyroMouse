function [t, q_true, omega_true, alpha_true, euler_true] = generate_trajectory()
% GENERATE_TRAJECTORY  Generate a smooth, known 3D rotation trajectory.
%
%   Produces a time-varying orientation defined by sinusoidal Euler angles
%   (ZYX convention), converts to quaternions, and numerically differentiates
%   to obtain angular velocity and angular acceleration in the body frame.
%
%   Outputs:
%       t           — (N×1) time vector [s]
%       q_true      — (N×4) unit quaternions [q0 q1 q2 q3], scalar-first
%       omega_true  — (N×3) angular velocity in body frame [rad/s]
%       alpha_true  — (N×3) angular acceleration in body frame [rad/s²]
%       euler_true  — (N×3) Euler angles [roll, pitch, yaw] in radians
%
%   Convention:
%       Euler order = ZYX (yaw → pitch → roll)
%       Quaternion  = scalar-first Hamilton [q0, q1, q2, q3]
%
%   University Project — Multi-IMU UKF Sensor Fusion
%   ---------------------------------------------------------------

    %% ================= Simulation Parameters =================
    fs       = 200;          % Sampling rate [Hz]
    dt       = 1 / fs;       % Time step [s]
    T        = 10;           % Duration [s]
    N        = T * fs + 1;   % Number of samples (include t=0)
    t        = (0:N-1)' * dt;  % Column time vector

    %% ================= Euler Angle Profiles ==================
    % Sinusoidal profiles — amplitudes in degrees, convert to radians.
    %   Yaw   ψ(t) = 30° sin(2π·0.5·t)
    %   Pitch θ(t) = 15° sin(2π·0.3·t + π/4)
    %   Roll  φ(t) = 10° sin(2π·0.2·t + π/2)

    yaw_deg   = 30;   f_yaw   = 0.5;   phase_yaw   = 0;
    pitch_deg = 15;   f_pitch = 0.3;   phase_pitch = pi/4;
    roll_deg  = 10;   f_roll  = 0.2;   phase_roll  = pi/2;

    yaw   = deg2rad(yaw_deg)   * sin(2*pi*f_yaw   * t + phase_yaw);
    pitch = deg2rad(pitch_deg) * sin(2*pi*f_pitch  * t + phase_pitch);
    roll  = deg2rad(roll_deg)  * sin(2*pi*f_roll   * t + phase_roll);

    % Store as [roll, pitch, yaw] for convenience
    euler_true = [roll, pitch, yaw];   % (N×3) in radians

    %% ================= Euler → Quaternion ====================
    % MATLAB's eul2quat expects [yaw pitch roll] for 'ZYX' and returns
    % scalar-first [q0 q1 q2 q3].
    q_true = eul2quat([yaw, pitch, roll], 'ZYX');  % (N×4)

    % Ensure quaternion continuity (no sign flips between consecutive steps)
    for k = 2:N
        if dot(q_true(k,:), q_true(k-1,:)) < 0
            q_true(k,:) = -q_true(k,:);
        end
    end

    %% ================= Angular Velocity ======================
    % The angular velocity in the body frame is obtained from:
    %       ω = 2 · q* ⊗ dq/dt
    % where ⊗ is the Hamilton quaternion product and q* is the conjugate.
    %
    % We first compute dq/dt with smooth finite differences, then extract ω.

    dqdt = zeros(N, 4);

    % Central difference for interior points
    dqdt(2:N-1, :) = (q_true(3:N, :) - q_true(1:N-2, :)) / (2*dt);
    % Forward difference at the first point
    dqdt(1, :) = (-3*q_true(1,:) + 4*q_true(2,:) - q_true(3,:)) / (2*dt);
    % Backward difference at the last point
    dqdt(N, :) = (3*q_true(N,:) - 4*q_true(N-1,:) + q_true(N-2,:)) / (2*dt);

    % ω = 2 * conj(q) ⊗ dq/dt   →   take the vector part
    %   conj(q) = [q0, -q1, -q2, -q3]
    omega_true = zeros(N, 3);
    for k = 1:N
        q_conj = [q_true(k,1), -q_true(k,2), -q_true(k,3), -q_true(k,4)];
        omega_quat = quatmultiply(q_conj, dqdt(k,:));
        % Result: [0, ωx, ωy, ωz] (scalar part ≈ 0 for unit quaternion)
        omega_true(k,:) = 2 * omega_quat(2:4);
    end

    %% ================= Angular Acceleration ==================
    % α = dω/dt in the body frame, again with smooth finite differences.

    alpha_true = zeros(N, 3);

    % Central difference for interior points
    alpha_true(2:N-1, :) = (omega_true(3:N, :) - omega_true(1:N-2, :)) / (2*dt);
    % Forward difference at the first point
    alpha_true(1, :) = (-3*omega_true(1,:) + 4*omega_true(2,:) - omega_true(3,:)) / (2*dt);
    % Backward difference at the last point
    alpha_true(N, :) = (3*omega_true(N,:) - 4*omega_true(N-1,:) + omega_true(N-2,:)) / (2*dt);

    %% ================= Sanity Checks =========================
    fprintf('generate_trajectory: %d samples, %.1f s, dt = %.4f s\n', N, T, dt);
    fprintf('  Euler range (deg): yaw [%.1f, %.1f], pitch [%.1f, %.1f], roll [%.1f, %.1f]\n', ...
        rad2deg(min(yaw)), rad2deg(max(yaw)), ...
        rad2deg(min(pitch)), rad2deg(max(pitch)), ...
        rad2deg(min(roll)), rad2deg(max(roll)));
    fprintf('  Max |ω| = %.4f rad/s,  Max |α| = %.4f rad/s²\n', ...
        max(vecnorm(omega_true, 2, 2)), max(vecnorm(alpha_true, 2, 2)));
end
