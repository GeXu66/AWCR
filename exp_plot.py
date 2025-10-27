import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
import multiprocessing


def get_output_label(output_name):
    """Return appropriate axis label based on output channel name"""
    labels = {
        'Fx': 'Longitudinal Force (N)',
        'Fy': 'Lateral Force (N)',
        'Fz': 'Vertical Force (N)',
        'Mx': 'Rolling Moment (N·m)',
        'My': 'Pitching Moment (N·m)',
        'Mz': 'Yawing Moment (N·m)'
    }
    return labels.get(output_name, output_name)


def plot_comparison(trad_predict, best_predict, real_out, condition_name, output_name, save_path=None):
    """
    Plot both traditional and best channel predictions against real values
    """
    # Get axis labels based on output channel
    y_label = get_output_label(output_name)

    # Convert lists to numpy arrays if they aren't already
    real_out = np.array(real_out)
    trad_predict = np.array(trad_predict)
    best_predict = np.array(best_predict)

    # Create figure with IEEE Transaction style
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    cut_range = (0.1, 0.8)
    start = int(len(real_out) * cut_range[0])
    end = int(len(real_out) * cut_range[1])
    real_out_cut = real_out[start:end]
    trad_predict_cut = trad_predict[start:end]
    best_predict_cut = best_predict[start:end]

    # Generate time points
    x = np.arange(1, len(real_out_cut) + 1)
    x = x / 500  # Convert to seconds

    # Define colors for consistent plotting
    color1, color2, color3 = '#f6541f', '#2299f0', '#72b607'

    # Plot data with distinctive line styles
    ax.plot(x, real_out_cut, label='Measured', color=color1, linewidth=2, linestyle='-')
    ax.plot(x, trad_predict_cut, label='Empirical Sensors', color=color2, linewidth=2, linestyle='--')
    ax.plot(x, best_predict_cut, label='Optimized Sensors', color=color3, linewidth=2, linestyle=':')

    # Set formatting for IEEE Transaction style
    ax.spines['top'].set_linewidth(1.1)
    ax.spines['right'].set_linewidth(1.1)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)

    # Add labels
    ax.set_xlabel('Time (Seconds)', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # Add legend in top-left corner
    # ax.legend(loc='upper left', fontsize=12, framealpha=1, edgecolor='black')
    ax.legend(
        loc='upper left',  # 图例自身的左上角作为定位锚点
        fontsize=12,
        bbox_to_anchor=(0, 1),  # 将图例左上角与 Axes 的 (0, 1) 对齐
        fancybox=False,  # 关闭圆角
        edgecolor='black',  # 设置边框颜色
        framealpha=1,  # 边框透明度为不透明
        borderaxespad=0  # 去掉与坐标轴的间距（可按需调整）
    )

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.close(fig)


def plot_combined_comparison(condition_data, output_names, condition_name, save_path=None):
    """
    Plot all six force/moment components in a single 6x1 figure
    """
    # Create a large figure with 6 subplots (6 rows, 1 column)
    fig, axes = plt.subplots(6, 1, figsize=(8, 12), constrained_layout=True)

    # Define colors for consistent plotting
    color1, color2, color3 = '#f6541f', '#2299f0', '#72b607'

    # Plot each force/moment component
    for i, (ax, output_name) in enumerate(zip(axes, output_names)):
        # Get data for this component
        key = f"{condition_name}-{output_name}"
        if key in condition_data:
            data = condition_data[key]

            real_data = np.array(data['real_values'])
            trad_data = np.array(data['traditional_predicted'])
            best_data = np.array(data['selected_predicted'])

            # Apply time range cut
            cut_range = (0.1, 0.8)
            start = int(len(real_data) * cut_range[0])
            end = int(len(real_data) * cut_range[1])
            real_data_cut = real_data[start:end]
            trad_data_cut = trad_data[start:end]
            best_data_cut = best_data[start:end]

            # Generate time points
            x = np.arange(1, len(real_data_cut) + 1)
            x = x / 500  # Convert to seconds

            # Get appropriate y-axis label
            y_label = get_output_label(output_name)

            # Plot the data
            ax.plot(x, real_data_cut, label='Measured', color=color1, linewidth=2, linestyle='-')
            ax.plot(x, trad_data_cut, label='Empirical Sensors', color=color2, linewidth=2, linestyle='--')
            ax.plot(x, best_data_cut, label='Optimized Sensors', color=color3, linewidth=2, linestyle=':')

            # Set formatting for IEEE Transaction style
            ax.spines['top'].set_linewidth(1.1)
            ax.spines['right'].set_linewidth(1.1)
            ax.spines['left'].set_linewidth(1.1)
            ax.spines['bottom'].set_linewidth(1.1)
            ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            # Add y-axis label
            ax.set_ylabel(y_label, fontsize=14)

            # Only add x-axis label to the bottom subplot
            if i == 5:  # Last subplot
                ax.set_xlabel('Time (Seconds)', fontsize=14)

            # Add legend to the first subplot only
            if i == 0:
                ax.legend(
                    loc='upper left',  # 图例自身的左上角作为定位锚点
                    fontsize=12,
                    bbox_to_anchor=(0, 1),  # 将图例左上角与 Axes 的 (0, 1) 对齐
                    fancybox=False,  # 关闭圆角
                    edgecolor='black',  # 设置边框颜色
                    framealpha=1,  # 边框透明度为不透明
                    borderaxespad=0  # 去掉与坐标轴的间距（可按需调整）
                )
    fig.align_ylabels(axes.ravel())

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
        print(f"Saved combined figure to {save_path}")

    plt.close(fig)


def generate_metric_summary(json_data, output_names):
    """Generate a summary of the metrics for all conditions"""
    # Initialize dictionaries to store metrics
    trad_metrics = {metric: [] for metric in ['rmse', 'mape', 'r2']}
    best_metrics = {metric: [] for metric in ['rmse', 'mape', 'r2']}

    # Collect all metrics
    for key, data in json_data.items():
        # Skip non-data keys if any
        if not isinstance(data, dict) or 'traditional_metrics' not in data:
            continue

        # Add traditional metrics
        for metric in trad_metrics:
            trad_metrics[metric].append(data['traditional_metrics'][metric])

        # Add best metrics
        for metric in best_metrics:
            best_metrics[metric].append(data['selected_metrics'][metric])

    # Convert to numpy arrays for calculation
    for metric in trad_metrics:
        trad_metrics[metric] = np.array(trad_metrics[metric])
        best_metrics[metric] = np.array(best_metrics[metric])

    # Calculate averages
    trad_avg = {metric: np.mean(values) for metric, values in trad_metrics.items()}
    best_avg = {metric: np.mean(values) for metric, values in best_metrics.items()}

    # Calculate improvement percentages
    improvements = {}
    improvements['rmse'] = ((trad_avg['rmse'] - best_avg['rmse']) / trad_avg['rmse']) * 100  # Lower is better
    improvements['mape'] = ((trad_avg['mape'] - best_avg['mape']) / trad_avg['mape']) * 100  # Lower is better
    # 对于 R², 这里假设 1 是“完美值”，因此基准是 (1 - trad_avg['r2'])
    improvements['r2'] = ((best_avg['r2'] - trad_avg['r2']) / (1 - trad_avg['r2'])) * 100  # Higher is better

    # Print summary
    print("\nMetric Summary:")
    print(f"{'Metric':<10} {'Traditional':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 55)
    print(f"{'RMSE':<10} {trad_avg['rmse']:<15.4f} {best_avg['rmse']:<15.4f} {improvements['rmse']:<15.2f}%")
    print(f"{'MAPE':<10} {trad_avg['mape']:<15.4f} {best_avg['mape']:<15.4f} {improvements['mape']:<15.2f}%")
    print(f"{'R²':<10} {trad_avg['r2']:<15.4f} {best_avg['r2']:<15.4f} {improvements['r2']:<15.2f}%")

    # Return the summary data
    return {
        'traditional': trad_avg,
        'optimized': best_avg,
        'improvements': improvements
    }


def generate_plots_for_condition(condition_name, data, output_names):
    """
    多进程的目标函数：为单个工况生成所有单通道和组合通道的图
    """
    print(f"[PID {multiprocessing.current_process().pid}] Generating plots for condition: {condition_name}")

    # 创建目录
    fig_dir = Path(f'./result/sensor_selection/fig/{condition_name}')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 逐通道绘图
    for output_name in output_names:
        key = f"{condition_name}-{output_name}"
        if key in data:
            real_values = data[key]['real_values']
            trad_predicted = data[key]['traditional_predicted']
            best_predicted = data[key]['selected_predicted']

            # 保存路径
            fig_path = fig_dir / f"{condition_name}-{output_name}.pdf"
            plot_comparison(
                trad_predict=trad_predicted,
                best_predict=best_predicted,
                real_out=real_values,
                condition_name=condition_name,
                output_name=output_name,
                save_path=fig_path
            )

            # 打印该通道的指标
            trad_metrics = data[key]['traditional_metrics']
            best_metrics = data[key]['selected_metrics']
            print(f"  {output_name} - Traditional RMSE: {trad_metrics['rmse']:.4f}, R²: {trad_metrics['r2']:.4f}")
            print(f"  {output_name} - Optimized  RMSE: {best_metrics['rmse']:.4f}, R²: {best_metrics['r2']:.4f}")

    # 绘制组合 6x1 图
    combined_fig_path = fig_dir / f"{condition_name}-combined.pdf"
    plot_combined_comparison(data, output_names, condition_name, combined_fig_path)


def plot_all_from_json(json_file):
    """Generate all plots from a JSON file, using multiprocessing for different conditions."""
    # 加载 JSON 数据
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 提取输出名称（通道）
    output_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    condition_names = set()

    for key in data.keys():
        if '-' in key:
            condition_name = key.split('-')[0]
            condition_names.add(condition_name)

    # 将每个工况的绘图任务放进多进程池
    args_list = [(cn, data, output_names) for cn in condition_names]

    # 可以根据机器实际CPU核数进行调整
    # 不指定时，默认进程数为系统 CPU 核心数
    with multiprocessing.Pool() as pool:
        pool.starmap(generate_plots_for_condition, args_list)

    # 全部绘图完成后，生成并打印整体指标汇总
    summary = generate_metric_summary(data, output_names)

    # 仅展示一个工况中的传统通道和优化通道信息（假定都相同）
    # 如果在 JSON 中的每个 key 下都有相同的 channels 信息，只要取第一个即可
    first_key = next(iter(data.keys()))
    traditional_channels = data[first_key]['traditional_channels']
    selected_channels = data[first_key]['selected_channels']

    print(f"\nTraditional Channels: {traditional_channels}")
    print(f"Selected Channels: {selected_channels}")


def plot_probability_distributions(probs, low_threshold=0.1, high_threshold=0.75):
    """
    绘制原始和重分配后的概率分布图。
    """
    categories = [f'WC {i + 1}' for i in range(len(probs))]

    # 绘制原始分布图
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    plt.xlim(-0.6, len(categories) - 0.4)
    plt.ylim(0, 1)
    plt.fill_between([-0.6, len(categories) - 0.4], 0, low_threshold, color='#bdc3c7', alpha=0.7)
    plt.axhline(y=low_threshold, color='#e74c3c', linestyle='--', alpha=1)
    plt.axhline(y=high_threshold, color='#e74c3c', linestyle='--', alpha=1)
    plt.title('Original Probability Distribution', fontsize=14)
    plt.ylabel('Probability', fontsize=12)
    plt.tick_params(axis='x', rotation=0)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    bars1 = plt.bar(categories, probs, color='#3498db')
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=13)
    plt.savefig('./images/original_distribution.png', dpi=300, bbox_inches='tight')

    # 概率重分配
    new_probs = probs.copy()
    new_probs[new_probs < low_threshold] = 0
    scaling_factor = 1 / np.sum(new_probs)
    new_probs = new_probs * scaling_factor

    # 绘制重分配后的分布图
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    plt.xlim(-0.6, len(categories) - 0.4)
    plt.ylim(0, 1)
    plt.fill_between([-0.6, len(categories) - 0.4], 0, low_threshold, color='#bdc3c7', alpha=0.7)
    plt.axhline(y=low_threshold, color='#e74c3c', linestyle='--', alpha=1)
    plt.axhline(y=high_threshold, color='#e74c3c', linestyle='--', alpha=1)
    plt.title('Redistributed Probability Distribution', fontsize=14)
    plt.ylabel('Probability', fontsize=12)
    plt.tick_params(axis='x', rotation=0)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    bars2 = plt.bar(categories, new_probs, color='#2ecc71')
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=13)
    plt.savefig('./images/redistributed_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 示例：直接调用（请把JSON路径替换为你的实际路径）
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    plot_all_from_json('./result/sensor_selection/data/BC30-WBMB.json')
