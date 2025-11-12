% EXP3 visualization script (not a function)
% Reads result/EXP3/combined.mat and draws a 3x1 figure:
% (1) Fz measured vs estimated
% (2) Condition recognition result over time (numeric IDs)
% (3) Error distribution of all six force components

clear; clc; close all;
set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultTextFontName', 'Times New Roman');

% Font controls
label_fontsize = 16;  % axis labels
axis_fontsize = 14;   % tick labels
legend_fontsize = 14; % legends

% Locate files
thisFile = mfilename('fullpath');
if isempty(thisFile)
    thisDir = fileparts(matlab.desktop.editor.getActiveFilename);
else
    thisDir = fileparts(thisFile);
end
resultDir = fullfile(thisDir, '..');
exp3Dir = fullfile(resultDir, 'EXP3');
dataPath = fullfile(exp3Dir, 'combined.mat');
if ~isfile(dataPath)
    error('combined.mat not found at %s', dataPath);
end
S = load(dataPath);
Y_true = double(S.Y_true);           % (N x 6)
Y_pred = double(S.Y_pred);           % (N' x 6)
class_ids = double(S.class_ids(:));  % ensure column vector
fs = double(S.fs);                   % sampling frequency
seg_len = double(S.seg_len);         % samples per segment
% Align lengths to avoid plotting mismatches
Ntrue = size(Y_true, 1);
Npred = size(Y_pred, 1);
Nid   = numel(class_ids);
N     = min([Ntrue, Npred, Nid]);
Y_true = Y_true(1:N, :);
Y_pred = Y_pred(1:N, :);
class_ids = class_ids(1:N);
t = (0:N-1) ./ fs;

% Figure canvas
fig = figure('Color','w','Position',[100 100 800 800]);
tl = tiledlayout(fig, 3, 2, 'TileSpacing','compact', 'Padding','compact');

% (1) Fz (assume column 1) - span full first row
nexttile([1 2]);
fz_idx = 1;
plot(t, Y_true(:, fz_idx), 'r-', 'LineWidth', 1.5); hold on;
plot(t, Y_pred(:, fz_idx), '--', 'Color', [0.45 0.71 0.03], 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', label_fontsize); ylabel('$F_z$', 'Interpreter', 'latex', 'FontSize', label_fontsize);
set(gca, 'FontSize', axis_fontsize); grid on; box on;
legend({'Measured','Estimated'}, 'Location','best', 'FontSize', legend_fontsize);
xlim([10, 15]);

% (2) Condition IDs over time (piecewise-constant) with markers every 5 s - span full second row
nexttile([1 2]);
% Downsample class_ids to per-segment ID for clearer plot, then expand as stairs
num_segs = max(1, floor(N / seg_len));
ids = zeros(1, num_segs);
for i = 1:num_segs
    s = (i-1)*seg_len + 1; e = min(N, i*seg_len);
    ids(i) = mode(class_ids(s:e));
end
tt = (0:num_segs) * (seg_len/fs);
if numel(ids) == 1
    hst = stairs([0, tt(end)], [ids(1), ids(1)], 'b-', 'LineWidth', 1.5);
else
    hst = stairs(tt, [ids, ids(end)], 'Color', [0.13 0.60 0.94], 'LineWidth', 1.5);
end
% Add markers at every 5s (segment boundary)
try
    set(hst, 'Marker', 'o', 'MarkerFaceColor', [0.13 0.60 0.94], 'MarkerEdgeColor', 'w', 'MarkerSize', 4, 'MarkerIndices', 1:numel(tt));
catch
    % Fallback: overlay scatter at boundaries
    hold on; scatter(tt, [ids(1), ids], 12, 'o', 'filled', 'MarkerFaceColor', [0.13 0.60 0.94], 'MarkerEdgeColor', 'w');
end
xlabel('Time (s)', 'FontSize', label_fontsize); ylabel('Condition ID', 'FontSize', label_fontsize);
set(gca, 'FontSize', axis_fontsize); grid on; box on;
xlim([0 max(t)]);
ylim([5 33]);
% Overlay colored segments for Unknown1..Unknown3 (IDs 26..28)
hold on;
known_max = 25;
uColors = [0.85 0.33 0.10; 0.49 0.18 0.56; 0.30 0.75 0.93];
for i = 1:num_segs
    if ids(i) >= known_max + 1 && ids(i) <= known_max + 3
        cIdx = ids(i) - known_max; % 1..3
        plot([tt(i) tt(i+1)], [ids(i) ids(i)], '-', 'Color', uColors(cIdx,:), 'LineWidth', 2.5);
        plot(tt(i), ids(i), 'o', 'MarkerFaceColor', uColors(cIdx,:), 'MarkerEdgeColor', 'w', 'MarkerSize', 4);
    end
end

% Text annotations for ranges
text(0.01, 0.93, '1-25 Known Conditions', 'Units', 'normalized', 'FontSize', axis_fontsize);
text(0.01, 0.83, '26-28 Unknown Conditions', 'Units', 'normalized', 'FontSize', axis_fontsize);

% (3) Error distributions split into Forces (F1-3) and Moments (M1-3)
err = Y_pred - Y_true; % signed residuals

% Left: Forces (first 3) - third row, first column
axL = nexttile;
colorsF = [0.23 0.49 0.77; 0.85 0.33 0.10; 0.47 0.67 0.19];
hold(axL, 'on');
for i = 1:3
    ei = err(:, i);
    ei = ei(~isnan(ei));
    if isempty(ei), continue; end
    [f, xi] = ksdensity(ei);
    plot(axL, xi, f, 'Color', colorsF(i,:), 'LineWidth', 1.8);
    % fill under curve with light alpha
    fill(axL, [xi, fliplr(xi)], [f, zeros(size(f))], colorsF(i,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    mu = mean(ei);
    xline(axL, mu, '--', 'Color', colorsF(i,:), 'LineWidth', 1.0);
end
title(axL, 'Forces', 'FontSize', label_fontsize);
xlabel(axL, 'Residual', 'FontSize', label_fontsize);
ylabel(axL, 'Density', 'FontSize', label_fontsize);
set(axL, 'FontSize', axis_fontsize); grid(axL, 'on'); box(axL, 'on');
legend(axL, {'F1','F2','F3'}, 'Location', 'northeast', 'FontSize', legend_fontsize);

% Right: Moments (last 3) - third row, second column
axR = nexttile;
colorsM = [0.55 0.34 0.29; 0.49 0.18 0.56; 0.30 0.75 0.93];
hold(axR, 'on');
for j = 1:3
    idx = 3 + j;
    ej = err(:, idx);
    ej = ej(~isnan(ej));
    if isempty(ej), continue; end
    [f, xi] = ksdensity(ej);
    plot(axR, xi, f, 'Color', colorsM(j,:), 'LineWidth', 1.8);
    fill(axR, [xi, fliplr(xi)], [f, zeros(size(f))], colorsM(j,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    mu = mean(ej);
    xline(axR, mu, '--', 'Color', colorsM(j,:), 'LineWidth', 1.0);
end
title(axR, 'Moments', 'FontSize', label_fontsize);
xlabel(axR, 'Residual', 'FontSize', label_fontsize);
ylabel(axR, 'Density', 'FontSize', label_fontsize);
set(axR, 'FontSize', axis_fontsize); grid(axR, 'on'); box(axR, 'on');
legend(axR, {'M1','M2','M3'}, 'Location', 'northeast', 'FontSize', legend_fontsize);

% Export
outDir = fullfile(fileparts(resultDir), 'images', 'exp3');
if ~exist(outDir, 'dir'), mkdir(outDir); end
exportgraphics(fig, fullfile(outDir, 'exp3.png'), 'Resolution', 600);
exportgraphics(fig, fullfile(outDir, 'exp3.pdf'), 'Resolution', 600);


