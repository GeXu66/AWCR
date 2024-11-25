import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt


def plot_probability_distributions(probs, low_threshold=0.1, high_threshold=0.75):
    """
    绘制原始和重分配后的概率分布图。

    Parameters:
    -----------
    probs : array-like
        原始概率分布
    low_threshold : float, optional
        低阈值, default=0.1
    high_threshold : float, optional 
        高阈值, default=0.75
    figsize : tuple, optional
        每张图的大小, default=(10, 5)
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
    # ax.grid(True, linestyle='--', linewidth=1)
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
    # ax.grid(True, linestyle='--', linewidth=1)
    bars2 = plt.bar(categories, new_probs, color='#2ecc71')
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=13)
    plt.savefig('./images/redistributed_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    # 使用示例
    if sys.platform.startswith('win'):
        matplotlib.use('TkAgg')
        plt.rcParams['font.family'] = 'Arial'
    # probs = np.array([0.35, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.04, 0.03])
    probs = np.array([0.35, 0.20, 0.08, 0.05, 0.04, 0.03, 0.04, 0.03, 0.01, 0.01])
    plot_probability_distributions(probs)
