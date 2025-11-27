% EXP4 visualization script
% Renders three 2x1 figures that compare Fz predictions with and without
% configuration changes for loading, tire, and suspension studies.

close all;
clearvars;

plotDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(plotDir, '..', 'EXP4');

titleFontSize = 16;
labelFontSize = 13;
legendFontSize = 11;
lineWidth = 1.8;

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');

measuredColor = [27, 158, 119] / 255;
baselineColor = [217, 95, 2] / 255;
changedColor = [117, 112, 179] / 255;

figInfo = { ...
    'loading_conditions', '(a) Effect of loading conditions on estimation error'; ...
    'tire_changes', '(b) Effect of tire changes'; ...
    'suspension_mods', '(c) Effect of suspension modifications' ...
};

for idx = 1:size(figInfo, 1)
    matName = sprintf('%s.mat', figInfo{idx, 1});
    matPath = fullfile(dataDir, matName);

    if ~isfile(matPath)
        warning('EXP4:MissingFile', 'File not found: %s. Skipping.', matPath);
        continue;
    end

    data = load(matPath);
    fig = figure('Color', 'w', 'Position', [100, 100, 800, 400]);

    subplot(2, 1, 1);
    plot(data.time_base, data.Fz_true_base, 'Color', measuredColor, 'LineWidth', lineWidth);
    hold on;
    plot(data.time_base, data.Fz_pred_base, 'Color', baselineColor, 'LineWidth', lineWidth);
    grid on;
    box on;
    ylabel('$F_z$', 'Interpreter', 'latex', 'FontSize', labelFontSize);
    title('No configuration change', 'FontSize', titleFontSize);
    legend({'Measured $F_z$', 'Predicted $F_z$'}, 'Interpreter', 'latex', ...
        'FontSize', legendFontSize, 'Location', 'best');

    subplot(2, 1, 2);
    plot(data.time_modified, data.Fz_true_modified, 'Color', measuredColor, 'LineWidth', lineWidth);
    hold on;
    plot(data.time_modified, data.Fz_pred_modified, 'Color', changedColor, 'LineWidth', lineWidth);
    grid on;
    box on;
    xlabel('Time (s)', 'FontSize', labelFontSize);
    ylabel('$F_z$', 'Interpreter', 'latex', 'FontSize', labelFontSize);
    title('With configuration change', 'FontSize', titleFontSize);
    legend({'Measured $F_z$', 'Predicted $F_z$'}, 'Interpreter', 'latex', ...
        'FontSize', legendFontSize, 'Location', 'best');

    sgtitle(figInfo{idx, 2}, 'FontSize', titleFontSize + 1);

    outName = sprintf('exp4_%s.png', figInfo{idx, 1});
    exportgraphics(fig, fullfile(plotDir, outName), 'Resolution', 300);
end
