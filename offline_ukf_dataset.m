% Offline UKF prototipi
% En yeni data/ukf_dataset_*.csv dosyasini okur, rest segmentlerinden R ve
% gyro bias baslangiclarini hesaplar, sonra 3 IMU verisini UKF ile fuse eder.
%
% State:
%   x = [q(4); omega(3); alpha(3); b_orta(3); b_sol(3); b_sag(3)]
%
% Measurement:
%   z = [gyro_orta; gyro_sol; gyro_sag; acc_sol-acc_orta; acc_sag-acc_orta]

clear;
clc;

data_dir = "data";
csv_path = "";        % Bos birakilirsa en yeni ukf_dataset_*.csv secilir.
imu_spacing_m = 0.03; % Orta sensore gore sol/sag mesafe. Gerekirse 0.02-0.04 arasi ayarla.

% Tuning:
% Raw gyro average'a daha yakin UKF icin gyro_R_scale kucuk,
% accdiff_R_scale buyuk tutulur.
gyro_R_scale = 1.0;
accdiff_R_scale = 1.0;

q_quat = 1e-7;
q_omega = 2e-3;
q_alpha = 2e-2;
q_bias = 1e-7;

p_quat = 1e-4;
p_omega = 5e-2;
p_alpha = 2e-1;
p_bias = 5e-3;

if csv_path == ""
    files = dir(fullfile(data_dir, "ukf_dataset_*.csv"));
    if isempty(files)
        error("data klasorunde ukf_dataset_*.csv bulunamadi.");
    end

    [~, newest_idx] = max([files.datenum]);
    csv_path = fullfile(files(newest_idx).folder, files(newest_idx).name);
end

fprintf("Dataset okunuyor: %s\n", csv_path);
T = readtable(csv_path, "TextType", "string");

t = T.t_pc_s;
labels = T.label;
N = height(T);

acc_orta = [T.orta_acc_x, T.orta_acc_y, T.orta_acc_z];
gyr_orta = [T.orta_gyro_x, T.orta_gyro_y, T.orta_gyro_z];
acc_sol  = [T.sol_acc_x,  T.sol_acc_y,  T.sol_acc_z];
gyr_sol  = [T.sol_gyro_x,  T.sol_gyro_y,  T.sol_gyro_z];
acc_sag  = [T.sag_acc_x,  T.sag_acc_y,  T.sag_acc_z];
gyr_sag  = [T.sag_gyro_x,  T.sag_gyro_y,  T.sag_gyro_z];

rest_mask = startsWith(labels, "rest");
if nnz(rest_mask) < 20
    error("R/bias hesaplamak icin yeterli rest segmenti yok.");
end

bias_orta0 = mean(gyr_orta(rest_mask, :), 1)';
bias_sol0  = mean(gyr_sol(rest_mask, :), 1)';
bias_sag0  = mean(gyr_sag(rest_mask, :), 1)';

fprintf("\nBaslangic gyro biaslari (rad/s):\n");
fprintf("orta = [% .6f % .6f % .6f]\n", bias_orta0);
fprintf("sol  = [% .6f % .6f % .6f]\n", bias_sol0);
fprintf("sag  = [% .6f % .6f % .6f]\n", bias_sag0);

acc_diff_sol0 = mean(acc_sol(rest_mask, :) - acc_orta(rest_mask, :), 1);
acc_diff_sag0 = mean(acc_sag(rest_mask, :) - acc_orta(rest_mask, :), 1);

fprintf("\nRest accelerometer fark offsetleri (m/s^2):\n");
fprintf("sol-orta = [% .6f % .6f % .6f]\n", acc_diff_sol0);
fprintf("sag-orta = [% .6f % .6f % .6f]\n", acc_diff_sag0);

Z_all = buildMeasurementMatrix(acc_orta, gyr_orta, acc_sol, gyr_sol, acc_sag, gyr_sag, ...
    acc_diff_sol0, acc_diff_sag0);
Z_rest = Z_all(rest_mask, :);

% Diagonal R daha kararlidir. Full covariance istersen diag(...) yerine cov(...) kullan.
R = diag(var(Z_rest, 0, 1) + 1e-8);
R(1:9, 1:9) = R(1:9, 1:9) * gyro_R_scale;
R(10:15, 10:15) = R(10:15, 10:15) * accdiff_R_scale;

fprintf("\nR diagonal ilk 9 gyro varyansi:\n");
disp(diag(R(1:9, 1:9))');
fprintf("R tuning: gyro_R_scale=%.3f, accdiff_R_scale=%.1f\n", gyro_R_scale, accdiff_R_scale);

% UKF ayarlari
L = 19;
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

x = zeros(L, 1);
x(1:4) = [1; 0; 0; 0];
x(5:7) = mean([gyr_orta(1, :)' - bias_orta0, gyr_sol(1, :)' - bias_sol0, gyr_sag(1, :)' - bias_sag0], 2);
x(8:10) = [0; 0; 0];
x(11:13) = bias_orta0;
x(14:16) = bias_sol0;
x(17:19) = bias_sag0;

P = diag([ ...
    p_quat * ones(1, 4), ...
    p_omega * ones(1, 3), ...
    p_alpha * ones(1, 3), ...
    p_bias * ones(1, 9)]);

Q = diag([ ...
    q_quat * ones(1, 4), ...
    q_omega * ones(1, 3), ...
    q_alpha * ones(1, 3), ...
    q_bias * ones(1, 9)]);

r_orta = [0; 0; 0];
r_sol = [0; imu_spacing_m; 0];
r_sag = [0; -imu_spacing_m; 0];

gyr_orta_corr0 = gyr_orta - bias_orta0';
gyr_sol_corr0 = gyr_sol - bias_sol0';
gyr_sag_corr0 = gyr_sag - bias_sag0';
omega_raw_avg = (gyr_orta_corr0 + gyr_sol_corr0 + gyr_sag_corr0) / 3;

x_hist = zeros(N, L);
innov_norm = zeros(N, 1);
nis = zeros(N, 1);
P_omega_diag_hist = zeros(N, 3);

fprintf("\nUKF calisiyor: %d ornek\n", N);

for k = 1:N
    if k == 1
        dt = median(diff(t));
    else
        dt = t(k) - t(k - 1);
        if ~isfinite(dt) || dt <= 0 || dt > 0.1
            dt = median(diff(t));
        end
    end

    X_sigma = sigmaPoints(x, P, L, lambda);
    X_pred = zeros(L, 2 * L + 1);

    for i = 1:(2 * L + 1)
        X_pred(:, i) = processModel(X_sigma(:, i), dt);
    end

    x_pred = weightedMeanState(X_pred, Wm);
    P_pred = Q;
    for i = 1:(2 * L + 1)
        dx = stateDiff(X_pred(:, i), x_pred);
        P_pred = P_pred + Wc(i) * (dx * dx');
    end
    P_pred = stabilizeCov(P_pred);

    Z_sigma = zeros(15, 2 * L + 1);
    for i = 1:(2 * L + 1)
        Z_sigma(:, i) = measurementModel(X_pred(:, i), r_orta, r_sol, r_sag);
    end

    z_pred = zeros(15, 1);
    for i = 1:(2 * L + 1)
        z_pred = z_pred + Wm(i) * Z_sigma(:, i);
    end

    P_zz = R;
    P_xz = zeros(L, 15);
    for i = 1:(2 * L + 1)
        dz = Z_sigma(:, i) - z_pred;
        dx = stateDiff(X_pred(:, i), x_pred);
        P_zz = P_zz + Wc(i) * (dz * dz');
        P_xz = P_xz + Wc(i) * (dx * dz');
    end
    P_zz = stabilizeCov(P_zz);

    z_actual = Z_all(k, :)';
    K = P_xz / P_zz;

    innovation = z_actual - z_pred;
    x = x_pred + K * innovation;
    x(1:4) = normalizeQuat(x(1:4));
    P = P_pred - K * P_zz * K';
    P = stabilizeCov(P);

    x_hist(k, :) = x';
    innov_norm(k) = norm(innovation);
    nis(k) = innovation' * (P_zz \ innovation);
    P_omega = P(5:7, 5:7);
    P_omega_diag_hist(k, :) = diag(P_omega)';

    if mod(k, 500) == 0
        fprintf("%d / %d\n", k, N);
    end
end

omega_ukf = x_hist(:, 5:7);
alpha_ukf = x_hist(:, 8:10);
bias_orta = x_hist(:, 11:13);
bias_sol = x_hist(:, 14:16);
bias_sag = x_hist(:, 17:19);

result_id = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
result_mat = fullfile(data_dir, "offline_ukf_result_" + result_id + ".mat");
result_csv = fullfile(data_dir, "offline_ukf_result_" + result_id + ".csv");

Result = table(T.sample_idx, t, labels, ...
    omega_ukf(:, 1), omega_ukf(:, 2), omega_ukf(:, 3), ...
    omega_raw_avg(:, 1), omega_raw_avg(:, 2), omega_raw_avg(:, 3), ...
    gyr_orta_corr0(:, 1), gyr_orta_corr0(:, 2), gyr_orta_corr0(:, 3), ...
    gyr_sol_corr0(:, 1), gyr_sol_corr0(:, 2), gyr_sol_corr0(:, 3), ...
    gyr_sag_corr0(:, 1), gyr_sag_corr0(:, 2), gyr_sag_corr0(:, 3), ...
    alpha_ukf(:, 1), alpha_ukf(:, 2), alpha_ukf(:, 3), ...
    innov_norm, nis, ...
    'VariableNames', {'sample_idx', 't_pc_s', 'label', ...
    'omega_ukf_x', 'omega_ukf_y', 'omega_ukf_z', ...
    'omega_raw_avg_x', 'omega_raw_avg_y', 'omega_raw_avg_z', ...
    'omega_orta_x', 'omega_orta_y', 'omega_orta_z', ...
    'omega_sol_x', 'omega_sol_y', 'omega_sol_z', ...
    'omega_sag_x', 'omega_sag_y', 'omega_sag_z', ...
    'alpha_ukf_x', 'alpha_ukf_y', 'alpha_ukf_z', ...
    'innovation_norm', 'nis'});

writetable(Result, result_csv);
save(result_mat, "x_hist", "omega_ukf", "omega_raw_avg", "alpha_ukf", ...
    "gyr_orta_corr0", "gyr_sol_corr0", "gyr_sag_corr0", ...
    "innov_norm", "nis", "P_omega_diag_hist", ...
    "bias_orta", "bias_sol", "bias_sag", "R", "Q", "P", "csv_path", ...
    "imu_spacing_m", "acc_diff_sol0", "acc_diff_sag0", ...
    "gyro_R_scale", "accdiff_R_scale", "q_quat", "q_omega", "q_alpha", "q_bias", ...
    "p_quat", "p_omega", "p_alpha", "p_bias");

fprintf("\nSonuc kaydedildi:\n");
fprintf("CSV: %s\n", result_csv);
fprintf("MAT: %s\n", result_mat);

plotResults(t, labels, omega_ukf, omega_raw_avg, alpha_ukf, innov_norm);
plotSensorFusionComparison(t, gyr_orta_corr0, gyr_sol_corr0, gyr_sag_corr0, ...
    omega_raw_avg, omega_ukf);
plotConsistencyTests(t, labels, nis);
printConsistencySummary(labels, nis);

function Z = buildMeasurementMatrix(acc_orta, gyr_orta, acc_sol, gyr_sol, acc_sag, gyr_sag, ...
    acc_diff_sol0, acc_diff_sag0)
    N = size(acc_orta, 1);
    Z = zeros(N, 15);

    Z(:, 1:3) = gyr_orta;
    Z(:, 4:6) = gyr_sol;
    Z(:, 7:9) = gyr_sag;
    Z(:, 10:12) = (acc_sol - acc_orta) - acc_diff_sol0;
    Z(:, 13:15) = (acc_sag - acc_orta) - acc_diff_sag0;
end

function X = sigmaPoints(x, P, L, lambda)
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

    for i = 1:(2 * L + 1)
        X(1:4, i) = normalizeQuat(X(1:4, i));
    end
end

function x_next = processModel(x, dt)
    q = normalizeQuat(x(1:4));
    omega = x(5:7);
    alpha = x(8:10);
    b_orta = x(11:13);
    b_sol = x(14:16);
    b_sag = x(17:19);

    omega_next = omega + alpha * dt;

    wx = omega_next(1);
    wy = omega_next(2);
    wz = omega_next(3);
    qw = q(1);
    qx = q(2);
    qy = q(3);
    qz = q(4);

    q_dot = 0.5 * [ ...
        -qx * wx - qy * wy - qz * wz;
         qw * wx - qz * wy + qy * wz;
         qz * wx + qw * wy - qx * wz;
        -qy * wx + qx * wy + qw * wz];

    q_next = normalizeQuat(q + q_dot * dt);

    x_next = [q_next; omega_next; alpha; b_orta; b_sol; b_sag];
end

function z = measurementModel(x, r_orta, r_sol, r_sag)
    omega = x(5:7);
    alpha = x(8:10);
    b_orta = x(11:13);
    b_sol = x(14:16);
    b_sag = x(17:19);

    gyro_orta = omega + b_orta;
    gyro_sol = omega + b_sol;
    gyro_sag = omega + b_sag;

    acc_orta = leverArmAccel(omega, alpha, r_orta);
    acc_sol = leverArmAccel(omega, alpha, r_sol);
    acc_sag = leverArmAccel(omega, alpha, r_sag);

    z = [gyro_orta; gyro_sol; gyro_sag; acc_sol - acc_orta; acc_sag - acc_orta];
end

function a = leverArmAccel(omega, alpha, r)
    a = cross(alpha, r) + cross(omega, cross(omega, r));
end

function x_mean = weightedMeanState(X, Wm)
    x_mean = X * Wm';
    x_mean(1:4) = normalizeQuat(x_mean(1:4));
end

function dx = stateDiff(x, x_ref)
    dx = x - x_ref;
    if dot(x(1:4), x_ref(1:4)) < 0
        dx(1:4) = -x(1:4) - x_ref(1:4);
    end
end

function q = normalizeQuat(q)
    n = norm(q);
    if n < 1e-12 || ~isfinite(n)
        q = [1; 0; 0; 0];
    else
        q = q / n;
    end
end

function P = stabilizeCov(P)
    P = 0.5 * (P + P');
    P = P + eye(size(P, 1)) * 1e-9;
end

function plotResults(t, labels, omega_ukf, omega_raw_avg, alpha_ukf, innov_norm)
    figure("Name", "Offline UKF - Angular Velocity");

    tiledlayout(4, 1);

    nexttile;
    plot(t, omega_raw_avg(:, 1), "Color", [0.65 0.65 0.65]); hold on;
    plot(t, omega_ukf(:, 1), "LineWidth", 1.1);
    grid on;
    ylabel("\omega_x");
    title("Sag-sol hareket ekseni");
    legend("Raw avg", "UKF");

    nexttile;
    plot(t, omega_raw_avg(:, 2), "Color", [0.65 0.65 0.65]); hold on;
    plot(t, omega_ukf(:, 2), "LineWidth", 1.1);
    grid on;
    ylabel("\omega_y");
    title("Yukari-asagi hareket ekseni");

    nexttile;
    plot(t, omega_raw_avg(:, 3), "Color", [0.65 0.65 0.65]); hold on;
    plot(t, omega_ukf(:, 3), "LineWidth", 1.1);
    grid on;
    ylabel("\omega_z");
    title("Roll ekseni");

    nexttile;
    plot(t, innov_norm, "LineWidth", 1.0);
    grid on;
    ylabel("Innovation");
    xlabel("t (s)");

    figure("Name", "Offline UKF - Angular Acceleration");
    plot(t, alpha_ukf, "LineWidth", 1.0);
    grid on;
    xlabel("t (s)");
    ylabel("\alpha (rad/s^2)");
    legend("\alpha_x", "\alpha_y", "\alpha_z");
    title("UKF angular acceleration estimate");

    unique_labels = unique(labels, "stable");
    fprintf("\nDataset segmentleri:\n");
    for i = 1:numel(unique_labels)
        fprintf("- %s\n", unique_labels(i));
    end
end

function plotSensorFusionComparison(t, gyr_orta, gyr_sol, gyr_sag, omega_raw_avg, omega_ukf)
    axis_names = ["x", "y", "z"];
    figure("Name", "Sensor Fusion Comparison - Single IMUs vs UKF");
    tiledlayout(3, 1);

    for axis_idx = 1:3
        nexttile;
        plot(t, gyr_orta(:, axis_idx), "Color", [0.70 0.70 0.70]); hold on;
        plot(t, gyr_sol(:, axis_idx), "Color", [0.45 0.65 0.95]);
        plot(t, gyr_sag(:, axis_idx), "Color", [0.70 0.55 0.85]);
        plot(t, omega_raw_avg(:, axis_idx), "k", "LineWidth", 1.0);
        plot(t, omega_ukf(:, axis_idx), "Color", [0.95 0.40 0.10], "LineWidth", 1.2);
        grid on;
        ylabel("\omega_" + axis_names(axis_idx));

        if axis_idx == 1
            title("Tek sensorler vs raw average vs UKF");
            legend("Orta", "Sol", "Sag", "Raw avg", "UKF");
        end

        if axis_idx == 3
            xlabel("t (s)");
        end
    end
end

function plotConsistencyTests(t, labels, nis)
    meas_dim = 15;

    [nis_low, nis_high] = chi2Bounds(meas_dim, 0.95);

    figure("Name", "UKF Consistency - NIS");

    plot(t, nis, "LineWidth", 1.0); hold on;
    yline(meas_dim, "--", "Expected");
    if isfinite(nis_low)
        yline(nis_low, ":", "95% low");
        yline(nis_high, ":", "95% high");
    end
    grid on;
    ylabel("NIS");
    xlabel("t (s)");
    title("Normalized Innovation Squared");

    unique_labels = unique(labels, "stable");
    for i = 1:numel(unique_labels)
        idx = find(labels == unique_labels(i), 1, "first");
        if ~isempty(idx)
            xline(t(idx), "Color", [0.45 0.45 0.45], "LineStyle", "-.");
        end
    end
end

function printConsistencySummary(labels, nis)
    fprintf("\nConsistency summary:\n");
    fprintf("NIS expected mean is measurement dimension m=15.\n");
    fprintf("NIS mean all samples: %.3f\n", mean(nis, "omitnan"));

    unique_labels = unique(labels, "stable");
    for i = 1:numel(unique_labels)
        mask = labels == unique_labels(i);
        fprintf("  NIS %-14s mean %.3f\n", unique_labels(i), mean(nis(mask), "omitnan"));
    end
end

function [low, high] = chi2Bounds(dof, confidence)
    alpha = 1 - confidence;
    if exist("chi2inv", "file") == 2
        low = chi2inv(alpha / 2, dof);
        high = chi2inv(1 - alpha / 2, dof);
    else
        low = NaN;
        high = NaN;
    end
end
