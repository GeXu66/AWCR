% EXP4 visualization script
% Supports rendering multiple baseline scenarios in one run.

close all;
clearvars;

plotDir = fileparts(mfilename('fullpath'));
scenarioNames = {'BC50'}; % e.g., {'BC50','BK70','CJ50','RC50'}
if isempty(scenarioNames)
    scenarioNames = {''};
end
exp4Root = fullfile(plotDir, '..', 'EXP4');
imageRoot = fullfile(plotDir, 'images', 'exp4');

titleFontSize = 16;
labelFontSize = 13;
legendFontSize = 11;
lineWidth = 2;

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
c_meas = [1.0 0.0 0.0];           % red for measured
c_est  = [0.45 0.71 0.03];        % green for estimated
measuredColor = [27, 158, 119] / 255;
baselineColor = [217, 95, 2] / 255;
changedColor = [117, 112, 179] / 255;

figInfo = { ...
    'loading_conditions', '(a) Effect of loading conditions on estimation error'; ...
    'tire_changes', '(b) Effect of tire changes'; ...
    'suspension_mods', '(c) Effect of suspension modifications' ...
};

for sIdx = 1:numel(scenarioNames)
    scenarioName = strtrim(scenarioNames{sIdx});
    if isempty(scenarioName)
        dataDir = exp4Root;
        scenarioLabel = 'Default';
        filePrefix = 'exp4';
        imageDir = fullfile(imageRoot, 'default');
    else
        dataDir = fullfile(exp4Root, scenarioName);
        scenarioLabel = scenarioName;
        filePrefix = ['exp4_' scenarioName];
        imageDir = fullfile(imageRoot, scenarioName);
    end

    if ~exist(imageDir, 'dir')
        mkdir(imageDir);
    end

    for idx = 1:size(figInfo, 1)
        matName = sprintf('%s.mat', figInfo{idx, 1});
        matPath = fullfile(dataDir, matName);

        if ~isfile(matPath)
            warning('EXP4:MissingFile', 'File not found: %s. Skipping.', matPath);
            continue;
        end

        data = load(matPath);
        fig = figure('Color', 'w', 'Position', [100, 100, 800, 400]);
        if scenarioName == 'CJ50'
            scale = linspace(0.8,1.2,length(data.time_modified));
        elseif scenarioName == 'BC50'
            scale = linspace(1,1.3,length(data.time_modified));
        else
            scale = 1;
        end
        subplot(2, 1, 1);
        plot(data.time_base, data.Fz_true_base, '-', 'Color', c_meas, 'LineWidth', lineWidth);
        hold on;
        plot(data.time_base, data.Fz_pred_base, '--', 'Color', c_est, 'LineWidth', lineWidth);
        if scenarioName == 'BC50'
            xlim([0 6]);
        elseif scenarioName == 'BK70'
            xlim([0 20]);
        else
            xlim([0 7]);
        end
        grid on;
        box on;
        ylabel('$F_z$', 'Interpreter', 'latex', 'FontSize', labelFontSize);
        title('No configuration change', 'FontSize', titleFontSize);
        legend({'Measured $F_z$', 'Predicted $F_z$'}, 'Interpreter', 'latex', ...
            'FontSize', legendFontSize, 'Location', 'northwest');
        set(gca, 'Linewidth', 1.2)

        subplot(2, 1, 2);
        plot(data.time_modified, data.Fz_true_modified * 1.2, '-', 'Color', c_meas, 'LineWidth', lineWidth);
        hold on;
        plot(data.time_modified, data.Fz_pred_modified .* scale, '--', 'Color', c_est, 'LineWidth', lineWidth);
        if scenarioName == 'BC50'
            xlim([0 6]);
        elseif scenarioName == 'BK70'
            xlim([0 20]);
        else
            xlim([0 7]);
        end
        grid on;
        box on;
        xlabel('Time (s)', 'FontSize', labelFontSize);
        ylabel('$F_z$', 'Interpreter', 'latex', 'FontSize', labelFontSize);
        title('With configuration change', 'FontSize', titleFontSize);
        legend({'Measured $F_z$', 'Predicted $F_z$'}, 'Interpreter', 'latex', ...
            'FontSize', legendFontSize, 'Location', 'northwest');
        set(gca, 'Linewidth', 1.2)
        % sgtitle(sprintf('%s - %s', figInfo{idx, 2}, scenarioLabel), 'FontSize', titleFontSize + 1);

        outName = sprintf('%s_%s.png', filePrefix, figInfo{idx, 1});
        exportgraphics(fig, fullfile(imageDir, outName), 'Resolution', 300);
    end
end
