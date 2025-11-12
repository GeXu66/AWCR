% EXP3 visualization script (not a function)
% Reads result/EXP3/combined.mat and draws a 3x1 figure:
% (1) Fz measured vs estimated
% (2) Condition recognition result over time (numeric IDs)
% (3) Error distribution of all six force components

clear; clc; close all;
set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultTextFontName', 'Times New Roman');

% Font controls
label_fontsize = 14;  % axis labels
axis_fontsize = 12;   % tick labels
legend_fontsize = 12; % legends

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
Y_true = S.Y_true;           % (N x 6)
Y_pred = S.Y_pred;           % (N x 6)
class_ids = S.class_ids;     % (N x 1) or (N,)
fs = double(S.fs);           % sampling frequency
seg_len = double(S.seg_len); % samples per segment
N = size(Y_true,1);
t = (0:N-1) ./ fs;

% Figure canvas
fig = figure('Color','w','Position',[100 100 800 800]);
tl = tiledlayout(fig, 3, 1, 'TileSpacing','compact', 'Padding','compact');

% (1) Fz (assume column 1)
nexttile;
fz_idx = 1;
plot(t, Y_true(:, fz_idx), 'r-', 'LineWidth', 1.2); hold on;
plot(t, Y_pred(:, fz_idx), '--', 'Color', [0.45 0.71 0.03], 'LineWidth', 1.2);
xlabel('Time (s)', 'FontSize', label_fontsize); ylabel('F_z', 'FontSize', label_fontsize);
set(gca, 'FontSize', axis_fontsize); grid on; box on;
legend({'Measured','Estimated'}, 'Location','northeast', 'FontSize', legend_fontsize);

% (2) Condition IDs over time (piecewise-constant)
nexttile;
% Downsample class_ids to per-segment ID for clearer plot, then expand as stairs
num_segs = floor(N / seg_len);
ids = zeros(1, num_segs);
for i = 1:num_segs
    s = (i-1)*seg_len + 1; e = i*seg_len;
    ids(i) = mode(class_ids(s:e));
end
tt = (0:num_segs) * (seg_len/fs);
stairs(tt, [ids, ids(end)], 'b-', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', label_fontsize); ylabel('Condition ID', 'FontSize', label_fontsize);
set(gca, 'FontSize', axis_fontsize); grid on; box on;

% (3) Error distribution across 6 outputs
nexttile;
err = Y_pred - Y_true; % signed residuals
% Use boxplot for distributions
boxplot(err, 'Labels', {'F1','F2','F3','F4','F5','F6'});
set(gca, 'FontSize', axis_fontsize);
xlabel('Force Components', 'FontSize', label_fontsize);
ylabel('Residual', 'FontSize', label_fontsize);
grid on; box on;

% Export
outDir = fullfile(fileparts(resultDir), 'images', 'exp3');
if ~exist(outDir, 'dir'), mkdir(outDir); end
exportgraphics(fig, fullfile(outDir, 'exp3.png'), 'Resolution', 600);
exportgraphics(fig, fullfile(outDir, 'exp3.pdf'), 'Resolution', 600);


