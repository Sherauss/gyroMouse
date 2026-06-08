

% UKF icin etiketli ham veri seti kaydedici
% ESP32 veri formati:
%   1-6   = Orta acc xyz, gyro xyz
%   7-12  = Sol  acc xyz, gyro xyz
%   13-18 = Sag  acc xyz, gyro xyz
%
% Cikti:
%   data/ukf_dataset_YYYYMMDD_HHMMSS.csv
%   data/ukf_dataset_YYYYMMDD_HHMMSS_meta.mat

clear;
clc;

portAdi = "COM7";
baudRate = 115200;

output_dir = "data";
if ~exist(output_dir, "dir")
    mkdir(output_dir);
end

session_id = string(datetime("now", "Format", "yyyyMMdd_HHmmss"));
csv_path = fullfile(output_dir, "ukf_dataset_" + session_id + ".csv");
mat_path = fullfile(output_dir, "ukf_dataset_" + session_id + "_meta.mat");

% Hareket protokolu. Sureleri ihtiyaca gore degistirebilirsin.
segments = [
    struct("label", "rest_initial",  "duration_s", 8,  "instruction", "Cihazi masada tamamen sabit tut.")
    struct("label", "yaw_x",         "duration_s", 12, "instruction", "Sadece saga-sola hareket ettir. Roll ve yukari-asagi karistirma.")
    struct("label", "rest_after_x",  "duration_s", 4,  "instruction", "Cihazi tekrar sabit tut.")
    struct("label", "pitch_y",       "duration_s", 12, "instruction", "Sadece yukari-asagi hareket ettir. Saga-sola ve roll karistirma.")
    struct("label", "rest_after_y",  "duration_s", 4,  "instruction", "Cihazi tekrar sabit tut.")
    struct("label", "roll_z",        "duration_s", 12, "instruction", "Sadece roll/burma hareketi yap.")
    struct("label", "rest_final",    "duration_s", 6,  "instruction", "Cihazi masada sabit tut.")
];

column_names = ["sample_idx", "t_pc_s", "segment_id", "label", ...
    "orta_acc_x", "orta_acc_y", "orta_acc_z", "orta_gyro_x", "orta_gyro_y", "orta_gyro_z", ...
    "sol_acc_x",  "sol_acc_y",  "sol_acc_z",  "sol_gyro_x",  "sol_gyro_y",  "sol_gyro_z", ...
    "sag_acc_x",  "sag_acc_y",  "sag_acc_z",  "sag_gyro_x",  "sag_gyro_y",  "sag_gyro_z"];

if exist('s', 'var')
    clear s;
end

s = serialport(portAdi, baudRate);
configureTerminator(s, "LF");
flush(s);

fprintf("Baglanti acildi: %s @ %d baud\n", portAdi, baudRate);
fprintf("Kayit dosyasi: %s\n\n", csv_path);
fprintf("Birimler: acc=m/s^2, gyro=rad/s\n");
fprintf("Sensor sirasi: Orta, Sol, Sag\n\n");

fprintf("Baslamak icin Enter'a bas. Sonra ekrandaki hareketleri yap.\n");
pause;

all_rows = {};
sample_idx = 0;
t0 = tic;

for segment_id = 1:numel(segments)
    label = segments(segment_id).label;
    duration_s = segments(segment_id).duration_s;
    instruction = segments(segment_id).instruction;

    fprintf("\n[%d/%d] %s\n", segment_id, numel(segments), label);
    fprintf("%s\n", instruction);
    countdown(3);

    segment_timer = tic;
    valid_count = 0;

    while toc(segment_timer) < duration_s
        veriDizisi = oku18(s);

        if isempty(veriDizisi)
            continue;
        end

        sample_idx = sample_idx + 1;
        valid_count = valid_count + 1;

        row = [{sample_idx, toc(t0), segment_id, char(label)}, num2cell(veriDizisi(:)')];
        all_rows(end + 1, :) = row; %#ok<SAGROW>
    end

    actual_duration = toc(segment_timer);
    fprintf("Segment bitti. Gecerli ornek: %d, yaklasik fs: %.1f Hz\n", ...
        valid_count, valid_count / actual_duration);
end

clear s;

T = cell2table(all_rows, "VariableNames", column_names);
writetable(T, csv_path);

meta = struct();
meta.session_id = session_id;
meta.portAdi = portAdi;
meta.baudRate = baudRate;
meta.sensor_order = "Orta, Sol, Sag";
meta.units = "acc=m/s^2, gyro=rad/s";
meta.columns = column_names;
meta.segments = segments;
meta.note = "Labels are user-performed motion classes, not precision ground truth angles.";

save(mat_path, "meta");

fprintf("\nKayit tamamlandi.\n");
fprintf("CSV : %s\n", csv_path);
fprintf("META: %s\n", mat_path);

function veriDizisi = oku18(s)
    strVeri = readline(s);
    veriDizisi = str2double(split(strtrim(strVeri), ","));

    if numel(veriDizisi) ~= 18 || any(~isfinite(veriDizisi))
        veriDizisi = [];
        return;
    end

    veriDizisi = veriDizisi(:);
end

function countdown(n)
    for k = n:-1:1
        fprintf("%d...\n", k);
        pause(1);
    end
    fprintf("Basla!\n");
end
