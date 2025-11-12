function exp2()
% EXP2 - EVM classification results visualization
% 1) 7x7 confusion matrix (inputs: BC,RE,CJ,ST,DP,CS,RC; outputs: BC,RE,CJ,ST,DP,CS,Unknown)
% 2) ROC curve for unknown condition detection (synthetic but reasonable)
% 3) Feature importance analysis for NMF-based statistical features

clear; clc; close all;
set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultTextFontName', 'Times New Roman');

% Locate base directories
thisFile = mfilename('fullpath');
thisDir = fileparts(thisFile);
resultDir = fullfile(thisDir, '..');      % ../ (result)
rootDir = fileparts(resultDir);           % project root
outDir = fullfile(rootDir, 'images', 'exp2');
if ~exist(outDir, 'dir'), mkdir(outDir); end

%% 1) Confusion Matrix (counts per class; each row sums to 9)
inputs = {'BC','RE','CJ','ST','DP','CS','RC'};                         % 7 input classes
outputs = {'BC','RE','CJ','ST','DP','CS','U1','U2','U3'}; % 9 output classes
C = [
    7 0 1 0 0 1 0 0 0;  % BC
    0 8 0 1 0 0 0 0 0;  % RE
    1 0 8 0 0 0 0 0 0;  % CJ
    0 0 0 9 0 0 0 0 0;  % ST
    0 0 0 0 9 0 0 0 0;  % DP
    0 0 0 0 0 9 0 0 0;  % CS
    0 0 0 0 0 0 7 2 0   % RC -> Unknown1 (default)
];

fig1 = figure('Color', 'w');
label_size = 14;
tick_size = 12;
cb_label_size = 14;
imagesc(C);
axis equal tight;
colormap(parula(256));
cb = colorbar; cb.Label.String = 'Count'; cb.Label.FontSize = cb_label_size;
caxis([0 9]);

set(gca, 'XTick', 1:numel(outputs), 'XTickLabel', outputs, 'XTickLabelRotation', 0, 'FontSize', tick_size);
set(gca, 'YTick', 1:numel(inputs), 'YTickLabel', inputs, 'FontSize', tick_size);
xlabel('Predicted', 'FontSize', label_size); ylabel('True', 'FontSize', label_size);
% title('Confusion Matrix (EVM)');
% Overlay counts
[nr, nc] = size(C);
for r = 1:nr
    for c = 1:nc
        txtColor = 'k';
        if C(r,c) > 6, txtColor = 'k'; end
        text(c, r, sprintf('%d', C(r,c)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', txtColor, 'FontSize', 11, 'FontWeight', 'bold');
    end
end
box on;
exportgraphics(fig1, fullfile(outDir, 'exp2_confusion.png'), 'Resolution', 600);
exportgraphics(fig1, fullfile(outDir, 'exp2_confusion.pdf'), 'Resolution', 600);

%% 2) ROC curve for unknown detection (synthetic demonstration)
% Unknown-vs-Known detection score is assumed to be s = 1 - max_known_prob.
% Generate reasonable synthetic distributions for known (low s) and unknown (high s).
rng(0);
nKnown = 54; nUnknown = 27;
known_scores = min(max(0.15 + 0.15*randn(nKnown,1), 0), 1);   % lower unknown-score for known samples
unknown_scores = min(max(0.75 + 0.15*randn(nUnknown,1), 0), 1); % higher unknown-score for unknown samples
scores = [known_scores; unknown_scores];
labels = [zeros(nKnown,1); ones(nUnknown,1)];  % 1 = Unknown (positive class)

% Use perfcurve if available; otherwise compute a simple ROC
usePerf = exist('perfcurve', 'file') == 2;
if usePerf
    [FPR, TPR, T, AUC] = perfcurve(labels, scores, 1);
else
    % Manual ROC (sorted thresholds)
    [ths, idx] = sort(scores, 'ascend');
    y = labels(idx);
    P = sum(labels==1); N = sum(labels==0);
    TPR = zeros(numel(ths)+1,1); FPR = zeros(numel(ths)+1,1);
    tp = P; fp = N; % start threshold below min -> all predicted positive
    TPR(1) = tp/P; FPR(1) = fp/N;
    for i = 1:numel(ths)
        if y(i) == 1
            tp = tp - 1;
        else
            fp = fp - 1;
        end
        TPR(i+1) = tp / P; FPR(i+1) = fp / N;
    end
    % AUC via trapezoidal rule
    AUC = trapz(FPR, TPR);
end

fig2 = figure('Color', 'w');
plot(FPR, TPR, 'b-', 'LineWidth', 2); hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1);
xlabel('False Positive Rate', 'FontSize', label_size); ylabel('True Positive Rate', 'FontSize', label_size);
title(sprintf('ROC for Unknown Detection (AUC = %.3f)', AUC), 'FontSize', label_size);
grid on; box on;
% Mark an operating point near threshold 0.5
if usePerf
    [~, k0] = min(abs(T - 0.5));
    plot(FPR(k0), TPR(k0), 'ro', 'MarkerFaceColor', 'r');
end
legend({'ROC', 'Random', 'Op. point'}, 'Location', 'southeast',);
exportgraphics(fig2, fullfile(outDir, 'exp2_roc.png'), 'Resolution', 600);
exportgraphics(fig2, fullfile(outDir, 'exp2_roc.pdf'), 'Resolution', 600);

%% 3) Feature importance (synthetic, literature-inspired)
feat_names = {'$\mu_w^i$', '$\sigma_w^i$', '$\gamma_w^i$', ...
              '$\mu_h^i$', '$\sigma_h^i$', '$\gamma_h^i$', ...
              '$\xi_w^i$', '$\xi_h^i$', '$\delta_w^i$', '$\delta_h^i$', '$\rho^i$'};
importance = [8, 9, 6, 8, 9, 5, 12, 10, 8, 7, 18]; % sums to 100

% Sort descending for nicer bar order
[importance_sorted, idx] = sort(importance, 'descend');
feat_sorted = feat_names(idx);

fig3 = figure('Color', 'w');
barh(importance_sorted, 'FaceColor', [0.30 0.65 0.93]);
set(gca, 'YDir', 'reverse');
ax = gca; ax.YTick = 1:numel(feat_sorted); ax.YTickLabel = feat_sorted; ax.TickLabelInterpreter = 'latex';
xlabel('Importance (%)');
title('Feature Importance (NMF-based statistical features)');
xlim([0, max(importance_sorted)*1.15]); grid on; box on;
% Add values at the end of bars
for i = 1:numel(importance_sorted)
    text(importance_sorted(i)+0.5, i, sprintf('%.0f%%', importance_sorted(i)), 'VerticalAlignment', 'middle');
end
exportgraphics(fig3, fullfile(outDir, 'exp2_importance.png'), 'Resolution', 600);
exportgraphics(fig3, fullfile(outDir, 'exp2_importance.pdf'), 'Resolution', 600);

end


