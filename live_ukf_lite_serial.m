

clearvars -except enable_mouse;
clc;

if ~exist("enable_mouse", "var")
    enable_mouse = false;
end

portAdi = "COM7";
baudRate = 115200;

calibration_s = 5;
calibration_hz_guess = 100;
num_calib_samples = calibration_s * calibration_hz_guess;

% UKF-lite tuning: omega changes fast, bias changes slowly.
q_omega = 1e-2;
q_bias = 1e-9;
r_gyro_scale = 0.8;

p_omega = 2e-2;
p_bias = 1e-4;

still_gyro_threshold = 0.05;
still_acc_threshold = 1.2;
omega_damping = 0.35;
bias_adapt_alpha = 0.002;

mouse_deadzone = 0.035;
mouse_gain_x = 18;
mouse_gain_y = -18;
max_mouse_step = 35;

fprintf("Canli UKF-lite seri port denemesi\n");
fprintf("Port: %s, baud: %d\n", portAdi, baudRate);
if enable_mouse
    fprintf("Mouse kontrolu: ACIK\n");
else
    fprintf("Mouse kontrolu: KAPALI\n");
end
fprintf("Cihazi masaya sabit koy. %d saniye kalibrasyon yapilacak.\n\n", calibration_s);

if exist("s", "var")
    clear s;
end

s = serialport(portAdi, baudRate);
configureTerminator(s, "LF");
flush(s);

try
    [bias_orta0, bias_sol0, bias_sag0, R] = calibrateGyroOnly(s, num_calib_samples, r_gyro_scale);

    fprintf("\nKalibrasyon tamamlandi.\n");
    fprintf("bias orta = [% .6f % .6f % .6f]\n", bias_orta0);
    fprintf("bias sol  = [% .6f % .6f % .6f]\n", bias_sol0);
    fprintf("bias sag  = [% .6f % .6f % .6f]\n", bias_sag0);
    fprintf("\nUKF-lite basladi. Durdurmak icin Ctrl+C.\n");

    mouse_robot = [];
    screen_w = 0;
    screen_h = 0;
    if enable_mouse
        try
            import java.awt.Robot;
            import java.awt.Toolkit;
            mouse_robot = Robot;
            screen_size = Toolkit.getDefaultToolkit().getScreenSize();
            screen_w = screen_size.getWidth();
            screen_h = screen_size.getHeight();
            fprintf("Mouse aktif. Cursor UKF-lite omega_x/y ile hareket edecek.\n");
        catch mouse_err
            enable_mouse = false;
            fprintf("Mouse baslatilamadi, sadece grafik devam edecek: %s\n", mouse_err.message);
        end
    end

    L = 12;
    alpha_ut = 1e-3;
    ki = 0;
    beta = 2;
    lambda = alpha_ut^2 * (L + ki) - L;

    Wm = zeros(1, 2 * L + 1);
    Wc = zeros(1, 2 * L + 1);
    Wm(1) = lambda / (L + lambda);
    Wc(1) = lambda / (L + lambda) + (1 - alpha_ut^2 + beta);
    Wm(2:end) = 1 / (2 * (L + lambda));
    Wc(2:end) = 1 / (2 * (L + lambda));

    % State: x = [omega(3); b_middle(3); b_left(3); b_right(3)].
    x = [zeros(3, 1); bias_orta0; bias_sol0; bias_sag0];
    % P is initial uncertainty, Q is process noise.
    P = diag([p_omega * ones(1, 3), p_bias * ones(1, 9)]);
    Q = diag([q_omega * ones(1, 3), q_bias * ones(1, 9)]);

    figure("Name", "Live UKF-lite - Angular Velocity");
    tiledlayout(4, 1);

    ax1 = nexttile;
    h_raw_x = animatedline("Color", [0.65 0.65 0.65]);
    h_ukf_x = animatedline("Color", [0.95 0.40 0.10], "LineWidth", 1.2);
    grid on; ylabel("\omega_x"); title("Sag-sol hareket"); legend("Raw avg", "UKF-lite");

    ax2 = nexttile;
    h_raw_y = animatedline("Color", [0.65 0.65 0.65]);
    h_ukf_y = animatedline("Color", [0.95 0.40 0.10], "LineWidth", 1.2);
    grid on; ylabel("\omega_y"); title("Yukari-asagi hareket");

    ax3 = nexttile;
    h_raw_z = animatedline("Color", [0.65 0.65 0.65]);
    h_ukf_z = animatedline("Color", [0.95 0.40 0.10], "LineWidth", 1.2);
    grid on; ylabel("\omega_z"); title("Roll");

    ax4 = nexttile;
    h_nis = animatedline("Color", [0.20 0.55 0.95], "LineWidth", 1.0);
    grid on; ylabel("NIS"); xlabel("t (s)"); title("NIS");

    linkaxes([ax1, ax2, ax3, ax4], "x");
    ylim(ax1, [-3 3]);
    ylim(ax2, [-3 3]);
    ylim(ax3, [-3 3]);
    ylim(ax4, [0 80]);

    t0 = tic;
    sample_count = 0;

    while true
        veri = readValid18(s);
        if isempty(veri)
            continue;
        end

        now_t = toc(t0);
        [orta_acc, orta_gyr, ~, sol_gyr, ~, sag_gyr] = splitSensors(veri);

        % Measurement vector: stacked gyro readings.
        z_actual = [orta_gyr; sol_gyr; sag_gyr];
        [x, P, nis] = ukfLiteStep(x, P, z_actual, Q, R, Wm, Wc, L, lambda);

        % Still detection for drift reduction.
        raw_avg = mean([orta_gyr - x(4:6), sol_gyr - x(7:9), sag_gyr - x(10:12)], 2);
        is_still = norm(raw_avg) < still_gyro_threshold && abs(norm(orta_acc) - 9.80665) < still_acc_threshold;

        if is_still
            x(1:3) = omega_damping * x(1:3);
            x(4:6) = (1 - bias_adapt_alpha) * x(4:6) + bias_adapt_alpha * orta_gyr;
            x(7:9) = (1 - bias_adapt_alpha) * x(7:9) + bias_adapt_alpha * sol_gyr;
            x(10:12) = (1 - bias_adapt_alpha) * x(10:12) + bias_adapt_alpha * sag_gyr;
        end

        % Estimated angular velocity used by the air mouse.
        omega_ukf = x(1:3);

        if enable_mouse
            [dx_mouse, dy_mouse] = omegaToMouseDelta(omega_ukf, mouse_deadzone, ...
                mouse_gain_x, mouse_gain_y, max_mouse_step);
            if dx_mouse ~= 0 || dy_mouse ~= 0
                moveMouse(mouse_robot, dx_mouse, dy_mouse, screen_w, screen_h);
            end
        end

        addpoints(h_raw_x, now_t, raw_avg(1));
        addpoints(h_ukf_x, now_t, omega_ukf(1));
        addpoints(h_raw_y, now_t, raw_avg(2));
        addpoints(h_ukf_y, now_t, omega_ukf(2));
        addpoints(h_raw_z, now_t, raw_avg(3));
        addpoints(h_ukf_z, now_t, omega_ukf(3));
        addpoints(h_nis, now_t, nis);

        if now_t > 20
            xlim(ax1, [now_t - 20, now_t]);
        end

        sample_count = sample_count + 1;
        if mod(sample_count, 10) == 0
            drawnow limitrate;
        end

        if mod(sample_count, 100) == 0
            fprintf("t=%6.2f omega=[% .3f % .3f % .3f] raw=[% .3f % .3f % .3f] still=%d NIS=% .2f\n", ...
                now_t, omega_ukf(1), omega_ukf(2), omega_ukf(3), ...
                raw_avg(1), raw_avg(2), raw_avg(3), is_still, nis);
        end
    end
catch err
    fprintf("\nUKF-lite durdu: %s\n", err.message);
    clear s;
    rethrow(err);
end

function [bias_orta, bias_sol, bias_sag, R] = calibrateGyroOnly(s, num_samples, r_gyro_scale)
    gyr_orta = zeros(num_samples, 3);
    gyr_sol = zeros(num_samples, 3);
    gyr_sag = zeros(num_samples, 3);

    count = 0;
    while count < num_samples
        veri = readValid18(s);
        if isempty(veri)
            continue;
        end

        [~, og, ~, sg, ~, rg] = splitSensors(veri);
        count = count + 1;

        gyr_orta(count, :) = og';
        gyr_sol(count, :) = sg';
        gyr_sag(count, :) = rg';

        if mod(count, 50) == 0
            fprintf("Kalibrasyon: %d / %d\n", count, num_samples);
        end
    end

    % Stationary calibration: initial gyro biases.
    bias_orta = mean(gyr_orta, 1)';
    bias_sol = mean(gyr_sol, 1)';
    bias_sag = mean(gyr_sag, 1)';

    % Measurement noise covariance from stationary gyro variance.
    Z = [gyr_orta, gyr_sol, gyr_sag];
    R = diag(var(Z, 0, 1) * r_gyro_scale + 1e-8);
end

function [x, P, nis] = ukfLiteStep(x, P, z_actual, Q, R, Wm, Wc, L, lambda)
    % Predict: random-walk process model.
    X_sigma = sigmaPoints(x, P, L, lambda);
    X_pred = X_sigma;

    x_pred = X_pred * Wm';
    P_pred = Q;
    for i = 1:(2 * L + 1)
        dx = X_pred(:, i) - x_pred;
        P_pred = P_pred + Wc(i) * (dx * dx');
    end
    P_pred = stabilizeCov(P_pred);

    % Map sigma points into measurement space.
    Z_sigma = zeros(9, 2 * L + 1);
    for i = 1:(2 * L + 1)
        Z_sigma(:, i) = measurementModelLite(X_pred(:, i));
    end

    z_pred = Z_sigma * Wm';
    P_zz = R;
    P_xz = zeros(L, 9);
    for i = 1:(2 * L + 1)
        dz = Z_sigma(:, i) - z_pred;
        dx = X_pred(:, i) - x_pred;
        P_zz = P_zz + Wc(i) * (dz * dz');
        P_xz = P_xz + Wc(i) * (dx * dz');
    end
    P_zz = stabilizeCov(P_zz);

    % Update: innovation, Kalman gain, corrected state.
    innovation = z_actual - z_pred;
    K = P_xz / P_zz;
    x = x_pred + K * innovation;
    P = P_pred - K * P_zz * K';
    P = stabilizeCov(P);
    % NIS monitors measurement consistency.
    nis = innovation' * (P_zz \ innovation);
end

function z = measurementModelLite(x)
    % Model: each gyro measures common omega plus its own bias.
    omega = x(1:3);
    b_orta = x(4:6);
    b_sol = x(7:9);
    b_sag = x(10:12);
    z = [omega + b_orta; omega + b_sol; omega + b_sag];
end

function veri = readValid18(s)
    str = readline(s);
    veri = str2double(split(strtrim(str), ","));
    if numel(veri) ~= 18 || any(~isfinite(veri))
        veri = [];
        return;
    end
    veri = veri(:);
end

function [orta_acc, orta_gyr, sol_acc, sol_gyr, sag_acc, sag_gyr] = splitSensors(veri)
    orta_acc = veri(1:3);
    orta_gyr = veri(4:6);
    sol_acc = veri(7:9);
    sol_gyr = veri(10:12);
    sag_acc = veri(13:15);
    sag_gyr = veri(16:18);
end

function X = sigmaPoints(x, P, L, lambda)
    % Unscented Transform: 2L+1 sigma points.
    P = stabilizeCov(P);
    jitter = 1e-9;
    for attempt = 1:5
        [S, ok] = chol((L + lambda) * P + eye(L) * jitter, "lower");
        if ok == 0
            break;
        end
        jitter = jitter * 10;
    end

    if ok ~= 0
        P = diag(max(diag(P), 1e-8));
        S = chol((L + lambda) * P, "lower");
    end

    X = zeros(L, 2 * L + 1);
    X(:, 1) = x;
    for i = 1:L
        X(:, i + 1) = x + S(:, i);
        X(:, i + 1 + L) = x - S(:, i);
    end
end

function P = stabilizeCov(P)
    P = 0.5 * (P + P');
    P = P + eye(size(P, 1)) * 1e-9;
end

function [dx, dy] = omegaToMouseDelta(omega, deadzone, gain_x, gain_y, max_step) %#ok<DEFNU>
    % Convert omega_x/y into cursor dx/dy.
    ox = omega(1);
    oy = omega(2);
    if abs(ox) < deadzone
        ox = 0;
    end
    if abs(oy) < deadzone
        oy = 0;
    end
    dx = round(gain_x * ox);
    dy = round(gain_y * oy);
    dx = max(min(dx, max_step), -max_step);
    dy = max(min(dy, max_step), -max_step);
end

function moveMouse(robot, dx, dy, screen_w, screen_h) %#ok<DEFNU>
    pointer = java.awt.MouseInfo.getPointerInfo().getLocation();
    x_new = pointer.getX() + dx;
    y_new = pointer.getY() + dy;
    x_new = max(min(x_new, screen_w - 1), 0);
    y_new = max(min(y_new, screen_h - 1), 0);
    robot.mouseMove(round(x_new), round(y_new));
end
