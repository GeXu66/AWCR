function exp1()
% EXP1 sequential learning performance demonstration (2x2 figure)
% Reads results under result/EXP1/<sample>/(update|unupdate)/prediction.mat
% Font: Times New Roman
clear;clc;close all;
set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultTextFontName', 'Times New Roman');

% Configuration
fz_idx = 1;              % Index of Fz in outputs (1-based)
fs = 500;                % Sampling frequency used in pipeline plots
% Annotations (user-specified sampling inclusion probabilities)
p_BC_first  = 0.692;     % For BC50_01
p_BC_second = 0.839;     % For BC50_02
p_RE_RC30   = 0.002;     % For RC30_01 recognized as RE fallback

% Locate result directory relative to this script
thisFile = mfilename('fullpath');
thisDir = fileparts(thisFile);
resultDir = fullfile(thisDir, '..');           % ../ (result)
expDir = fullfile(resultDir, 'EXP1');

% Sample names
s1 = 'BC50_01';
s2 = 'BC50_02';
s3 = 'RC30_01';

% Load data (prefers update/, falls back to unupdate/)
[Yt1, Yp1, dir1] = load_pred(expDir, s1);
[Yt2, Yp2, dir2] = load_pred(expDir, s2);
[Yt3, Yp3, dir3] = load_pred(expDir, s3);

% Build time axes
t1 = (0:size(Yt1,1)-1) ./ fs;
t2 = (0:size(Yt2,1)-1) ./ fs;
t3 = (0:size(Yt3,1)-1) ./ fs;

% Compute errors for panel (c)
e1 = abs(Yp1(:, fz_idx) - Yt1(:, fz_idx));
e2 = abs(Yp2(:, fz_idx) - Yt2(:, fz_idx));

% Read MAPE from metrics.txt if available; otherwise compute
mape1 = read_mape(expDir, s1, dir1);
if isnan(mape1)
    mape1 = 100.0 * mean(abs((Yp1(:, fz_idx) - Yt1(:, fz_idx)) ./ max(abs(Yt1(:, fz_idx)), eps)));
end
mape2 = read_mape(expDir, s2, dir2);
if isnan(mape2)
    mape2 = 100.0 * mean(abs((Yp2(:, fz_idx) - Yt2(:, fz_idx)) ./ max(abs(Yt2(:, fz_idx)), eps)));
end

label_fontsize = 12;
legend_fontsize = 10;
axis_fontsize = 10;

% Figure and panels
fig = figure('Color', 'w', 'Position', [500 500 600*1.1 300*1.1]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
% title(tl, 'Sequential learning performance demonstration.')

% (a) First encounter with BC50
nexttile;
plot(t1, Yt1(:, fz_idx), 'Color', [0.13 0.60 0.94], 'LineWidth', 1.5); hold on;
plot(t1, Yp1(:, fz_idx), '--', 'Color', [0.45 0.71 0.03], 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', label_fontsize); ylabel('$F_z$', 'Interpreter', 'latex', 'FontSize', label_fontsize); 
legend({'Measured', 'Estimated'}, 'Location', 'southeast', 'FontSize', legend_fontsize);
text(0.02, 0.95, sprintf('p(BC)=%.3f', p_BC_first), 'Units', 'normalized', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
xlim([0, max(t1)]);
% title('(a) Estimation performance for F_z during the first encounter with BC50');

% (b) Second encounter with BC50
nexttile;
plot(t2, Yt2(:, fz_idx), 'Color', [0.13 0.60 0.94], 'LineWidth', 1.5); hold on;
plot(t2, Yp2(:, fz_idx), '--', 'Color', [0.45 0.71 0.03], 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', label_fontsize); ylabel('$F_z$', 'Interpreter', 'latex', 'FontSize', label_fontsize); 
legend({'Measured', 'Estimated'}, 'Location', 'southeast', 'FontSize', legend_fontsize);
text(0.02, 0.95, sprintf('p(BC)=%.3f', p_BC_second), 'Units', 'normalized', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
xlim([0, max(t2)]);
% title('(b) Estimation performance for F_z during the second encounter with BC50');

% (c) Error comparison first vs second
nexttile;
plot(t1, e1, 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2); hold on;
plot(t2, e2, 'Color', [0.49 0.18 0.56], 'LineWidth', 1.2);
xlabel('Time (s)', 'FontSize', label_fontsize); ylabel('|Error|', 'FontSize', label_fontsize); 
legend({'BC50\_01', 'BC50\_02'}, 'Location', 'northwest', 'FontSize', legend_fontsize);
xlim([0, max(t1)]);
% text(0.02, 0.95, sprintf('MAPE(BC50\\_01)=%.1f%%, MAPE(BC50\\_02)=%.1f%%', mape1, mape2), 'Units', 'normalized', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
% title('(c) Comparison of initial error (1st) vs adapted error (2nd)');

% (d) Novel condition RC30 (first encounter)
nexttile;
plot(t3, Yt3(:, fz_idx), 'Color', [0.13 0.60 0.94], 'LineWidth', 1.5); hold on;
plot(t3, Yp3(:, fz_idx), '--', 'Color', [0.45 0.71 0.03], 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', label_fontsize); ylabel('$F_z$', 'Interpreter', 'latex', 'FontSize', label_fontsize); legend({'Measured', 'Estimated'}, 'Location', 'northeast', 'FontSize', legend_fontsize);
text(0.02, 0.95, sprintf('p(RE)=%.3f', p_RE_RC30), 'Units', 'normalized', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
% title('(d) Adaptation process when encountering the novel condition RC30');

end

function [Y_true, Y_pred, modeDir] = load_pred(expDir, sampleName)
% Try update/ first, then unupdate/
modeDir = 'update';
matPath = fullfile(expDir, sampleName, modeDir, 'prediction.mat');
if ~isfile(matPath)
    modeDir = 'unupdate';
    matPath = fullfile(expDir, sampleName, modeDir, 'prediction.mat');
end
if ~isfile(matPath)
    error('prediction.mat not found for %s in update/ or unupdate/.', sampleName);
end
S = load(matPath);
if isfield(S, 'Y_true') && isfield(S, 'Y_pred')
    Y_true = S.Y_true;
    Y_pred = S.Y_pred;
else
    error('prediction.mat for %s does not contain Y_true/Y_pred.', sampleName);
end
end

function mape = read_mape(expDir, sampleName, modeDir)
% Read MAPE from metrics.txt (if exists). Returns NaN if not present.
mape = NaN;
metricsPath = fullfile(expDir, sampleName, modeDir, 'metrics.txt');
if ~isfile(metricsPath)
    return;
end
try
    txt = fileread(metricsPath);
    % Expect line: MAPE: <value>
    expr = 'MAPE:\s*([0-9\.Ee\-\+]+)';
    tokens = regexp(txt, expr, 'tokens', 'once');
    if ~isempty(tokens)
        mape = str2double(tokens{1}) * 100.0; % convert to percentage
    end
catch
    % ignore parse errors
end
end


