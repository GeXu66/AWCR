import os
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy.io as scio
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import NMF
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from matplotlib import cm
from DataHeader import WC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import glob
from utils import plot_feature_distributions, plot_feature_distributions_single
from scipy.fftpack import fft, fftfreq
import numpy as np
import matplotlib as mpl
import json


def read_mat(filename):
    train = []
    target = []
    for file in filename:
        key = file.split('/')[-1]
        key = key.split('.')[-2]
        data = scio.loadmat(file)
        data = data[key]
        train.append(np.hstack((data[:, 90:106], data[:, 109:112], data[:, 118:121])))
        # train.append(np.hstack((data[:, 4:8], data[:, 56:138])))
        target.append(np.reshape(data[:, 34], (-1, 1)))
        # print(f'data {key} load successfully!')
    train_all = train[0]
    target_all = target[0]
    for i in range(1, len(train)):
        train_all = np.vstack((train_all, train[i]))
        target_all = np.vstack((target_all, target[i]))
    return train_all, target_all


def read_mat_split(filename):
    key = filename.split('/')[-1]
    key = key.split('.')[-2]
    data = scio.loadmat(filename)
    data = data[key]
    train = np.hstack((data[:, 90:106], data[:, 109:112], data[:, 118:121]))
    # train = np.hstack((data[:, 4:8], data[:, 56:138]))
    target = np.reshape(data[:, 34], (-1, 1))
    # print(f'data {key} load successfully!')
    return train, target


def compute_stft(signal_name):
    trainFile = ['../Data/matdata/' + signal_name + '.mat']
    X, y = read_mat(trainFile)
    pca = PCA(n_components=1)
    # 标准化数据，PCA对数据的缩放很敏感，所以通常需要先进行标准化
    sensor_scaler = StandardScaler()
    X_scaled = sensor_scaler.fit_transform(X)
    X_reduced = pca.fit_transform(X_scaled)
    x = np.squeeze(X_reduced, 1)
    y = np.squeeze(y, 1)
    # 示例信号
    fs = 512  # 采样频率
    # 计算短时傅里叶变换
    f, t, z = stft(x, fs, nperseg=32)
    return fs, f, t, z, x, y


def compute_stft_split(signal_name):
    X, y = read_mat_split(signal_name)
    pca = PCA(n_components=1)
    # 标准化数据，PCA对数据的缩放很敏感，所以通常需要先进行标准化
    sensor_scaler = StandardScaler()
    X_scaled = sensor_scaler.fit_transform(X)
    X_reduced = pca.fit_transform(X_scaled)
    x = np.squeeze(X_reduced, 1)
    y = np.squeeze(y, 1)
    # 示例信号
    fs = 512  # 采样频率
    # 计算短时傅里叶变换
    f, t, z = stft(x, fs, nperseg=32)
    return fs, f, t, z, x, y


def figure_stft():
    name1 = "DP30_01"
    name2 = "BR20_01"
    file_name = [name1, name2]
    fs, f1, t1, z1, input1, y1 = compute_stft(file_name[0])
    _, f2, t2, z2, input2, y2 = compute_stft(file_name[1])
    plt.style.use('ggplot')
    fig = plt.figure(layout="constrained", figsize=(13, 7))
    # fig, (ax1, ax2) = plt.subplots(2, 2, gridspec_kw={'height_ratios': [1, 1.7]}, figsize=(10, 8))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1.8], width_ratios=[1, 0.1, 1, 0.1])
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:])
    ax3 = fig.add_subplot(gs[-1, 0])
    ax3_colorbar = fig.add_subplot(gs[-1, 1])
    ax4 = fig.add_subplot(gs[-1, 2])
    ax4_colorbar = fig.add_subplot(gs[-1, 3])
    # 第一个子图：原信号
    ax1.plot(np.arange(len(input1)) / fs, input1, 'C0', label=name1.split('_')[0], linewidth=1.2)  # 假设y是时间序列数据
    # ax1.set_title('Original Signal')
    ax1.set_ylabel('Force (N)', fontsize=15)
    ax1.set_xlabel('Time (seconds)', fontsize=15)
    ax1.legend(fontsize=15, loc='upper right')
    ax1.grid(True)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax2.plot(np.arange(len(input2)) / fs, input2, 'C0', label=name2.split('_')[0], linewidth=1.2)  # 假设y是时间序列数据
    # ax2.set_title('Original Signal')
    ax2.set_ylabel('Force (N)', fontsize=15)
    ax2.set_xlabel('Time (seconds)', fontsize=15)
    ax2.legend(fontsize=15, loc='upper right')
    ax2.grid(True)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    pcm = ax3.pcolormesh(t1, f1, np.abs(z1), shading='gouraud')
    ax3.set_ylim([0, 40])
    # ax3.set_title('STFT Magnitude')
    ax3.set_ylabel('Frequency (Hz)', fontsize=15)
    ax3.set_xlabel('Time (seconds)', fontsize=15)
    plt.colorbar(pcm, cax=ax3_colorbar)
    ax3_colorbar.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    pcm = ax4.pcolormesh(t2, f2, np.abs(z2), shading='gouraud')
    ax4.set_ylim([0, 40])
    # ax4.set_title('STFT Magnitude')
    ax4.set_ylabel('Frequency (Hz)', fontsize=15)
    ax4.set_xlabel('Time (seconds)', fontsize=15)
    plt.colorbar(pcm, cax=ax4_colorbar)
    ax4_colorbar.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.rcParams['figure.constrained_layout.use'] = True
    plt.show()
    fig.savefig("images/TF.pdf", dpi=600)
    # plt.style.use('default')
    # X, Y = np.meshgrid(t2, f2)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, np.abs(z2), cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # plt.show()


def segment_figure():
    name = "RE40_01"
    x_len = 8000
    # name = "BR40_01"
    # x_len = 10000
    _, f, t, z, input_data, y = compute_stft(name)
    fig, ax = plt.subplots(1, layout="constrained", figsize=(14, 3))
    Tw = 5 * 512
    input_data = input_data[2000:]

    seg_num = 4
    overlap = int(Tw - (x_len - Tw) / (seg_num - 1)) + 1
    shift = Tw - overlap
    # 绘制曲线
    ax.plot(input_data, color='C0', linewidth=1.5, linestyle='-')
    ax.set_xlim([0, x_len])
    ax.grid(False)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    # ax.set_xlabel('Samples', fontsize=14)
    # ax.set_ylabel('Force (N)', fontsize=14)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    for i in range(seg_num):
        ax.fill_between(np.arange(0 + i * shift, Tw - 1 + i * shift), 0, 1, color=f'C{i}', alpha=0.3, transform=ax.get_xaxis_transform())
        fig_sub, ax_sub = plt.subplots(1, layout="constrained", figsize=(7, 1.5))
        ax_sub.plot(input_data[0 + i * shift:Tw - 1 + i * shift], color='C0', linewidth=1.5, linestyle='-')
        ax_sub.fill_between(np.arange(0, Tw), 0, 1, color=f'C{i}', alpha=0.3, transform=ax_sub.get_xaxis_transform())
        ax_sub.set_xlim([0, Tw])
        ax_sub.yaxis.set_ticklabels([])
        ax_sub.yaxis.set_ticks([])
        ax_sub.xaxis.set_ticklabels([])
        # fig_sub.savefig(f"images/{name}_Segment_sub_{i}.png", dpi=300)
    # ax.fill_between(np.arange(0, 2560), 0, 1, color='C1', alpha=0.3, transform=ax.get_xaxis_transform(), label='segment 1')
    # ax.fill_between(np.arange(1280, 3840), 0, 1, color='C2', alpha=0.3, transform=ax.get_xaxis_transform(), label='segment 2')
    # ax.fill_between(np.arange(2440, 5000), 0, 1, color='C3', alpha=0.3, transform=ax.get_xaxis_transform(), label='segment 3')
    # ax.annotate('', xy=(0, 6), xytext=(2560, 6), arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    # ax.text(1280, 5.3, 'Segment 1', color='black', fontsize=20, ha='center')
    #
    # ax.annotate('', xy=(1280, 5), xytext=(3840, 5), arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    # ax.text(2560, 4.3, 'Segment 2', color='black', fontsize=20, ha='center')
    #
    # ax.annotate('', xy=(2440, -5), xytext=(5000, -5), arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    # ax.text(3720, -5.7, 'Segment 3', color='black', fontsize=20, ha='center')
    # 显示图形
    plt.show()
    # fig.savefig(f"images/{name}_Segment.png", dpi=600)


def segment_fusion_figure():
    Tw = 3 * 512
    name1 = "CR15_03"
    _, f, t, z, input_data, y = compute_stft(name1)
    fusion_data = input_data[500:3500].reshape(1, -1)
    tick1 = 3000

    name2 = "CS20_03"
    _, f, t, z, input_data, y = compute_stft(name2)
    input_data = input_data[1000:4000].reshape(1, -1)
    fusion_data = np.hstack((fusion_data, input_data))
    tick2 = tick1 + 3000

    name3 = "BC30_01"
    _, f, t, z, input_data, y = compute_stft(name3)
    input_data = input_data[500:6 * Tw - tick2 + 500].reshape(1, -1)
    fusion_data = np.hstack((fusion_data, input_data))
    tick3 = tick2 + 3000

    x_len = fusion_data.shape[1]
    fusion_data = fusion_data.squeeze(0)
    fig, ax = plt.subplots(1, layout="constrained", figsize=(14, 3))

    seg_num = 6
    overlap = int(Tw - (x_len - Tw) / (seg_num - 1)) + 1
    # overlap = 0
    shift = Tw - overlap
    # 绘制曲线
    ax.plot(fusion_data, color='C0', linewidth=3, linestyle='-')
    ax.set_xlim([0, x_len])
    ax.grid(False)
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    ax.set_xticks([])
    # ax.set_xlabel('Samples', fontsize=14)
    # ax.set_ylabel('Force (N)', fontsize=14)
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    for i in range(seg_num):
        ax.fill_between(np.arange(0 + i * shift, Tw - 1 + i * shift), 0, 1, color=f'C{i}', alpha=0.3, transform=ax.get_xaxis_transform())
        fig_sub, ax_sub = plt.subplots(1, layout="constrained", figsize=(9, 1))
        ax_sub.plot(fusion_data[0 + i * shift:Tw - 1 + i * shift], color='C0', linewidth=3, linestyle='-')
        ax_sub.fill_between(np.arange(0, Tw), 0, 1, color=f'C{i}', alpha=0.3, transform=ax_sub.get_xaxis_transform())
        ax_sub.set_xlim([0, Tw])
        ax_sub.yaxis.set_ticklabels([])
        ax_sub.yaxis.set_ticks([])
        ax_sub.xaxis.set_ticklabels([])
        ax_sub.set_xticks([])
        fig_sub.savefig(f"images/{name1}_{name2}_{name3}_Segment_sub_{i}.png", dpi=300)
    # 显示图形
    plt.show()
    fig.savefig(f"images/{name1}_{name2}_{name3}_Segment.png", dpi=600)


def compute_tf_feature(name, split):
    if split:
        fs, f, t, z, x, y = compute_stft_split(name)
    else:
        fs, f, t, z, x, y = compute_stft(name)
    trainFile = ['../Data/matdata/' + name + '.mat']
    X, y = read_mat(trainFile)
    # pca = PCA(n_components=1)
    # sensor_scaler = StandardScaler()
    # X_scaled = sensor_scaler.fit_transform(X)
    # X = pca.fit_transform(X_scaled)
    # x = np.squeeze(X_reduced, 1)
    kurt_time = np.median([pd.Series(np.squeeze(x)).kurt() for x in X.T])
    skewness_time = np.median([pd.Series(np.squeeze(x)).skew() for x in X.T])
    # periodic
    fft_series = [fft(col) for col in X.T]
    power = [np.abs(fft_col) for fft_col in fft_series]
    sample_freq = [fftfreq(fft_col.size) for fft_col in fft_series]
    # 取top-3振幅值对应的周期作为周期性特征
    top_k_seasons = 1
    fft_periods = []
    for power_i, freqs_i in zip(power, sample_freq):
        # 找到振幅值最大的top_k_seasons个对应的索引
        top_k_indices = np.argsort(power_i)[-top_k_seasons:]
        # 使用这些索引找到对应的频率
        top_k_freqs = freqs_i[top_k_indices]
        # 计算周期性特征，即1/频率
        periods = [1 / freq if freq != 0 else 0 for freq in top_k_freqs]
        fft_periods.append(periods)
    fft_periods_max = np.median(abs(np.array(fft_periods)))

    abs_z = np.abs(z)
    # 设置 NMF 模型的参数
    n_components = 1  # 分解后的矩阵的列数
    model = NMF(n_components=n_components, init='random', random_state=0)
    # 对时频矩阵进行 NMF 分解
    W = model.fit_transform(abs_z)  # W 是系数矩阵
    H = model.components_  # H 是基矩阵
    rows = W.shape[0]
    cols = H.shape[1]
    sparsity_coff_vector = (np.sqrt(rows) - np.sum(W) / np.sum(np.squeeze(W) ** 2)) / (np.sqrt(rows) - 1)
    sparsity_base_vector = (np.sqrt(cols) - np.sum(H) / np.sum(np.squeeze(H) ** 2)) / (np.sqrt(cols) - 1)
    discontinuity_coff_vector = np.linalg.norm(np.diff(np.squeeze(W)), 2)
    discontinuity_base_vector = np.linalg.norm(np.diff(np.squeeze(H)), 2)
    alpha = 2
    # 对矩阵进行归一化
    normalized_z = abs_z / np.sum(abs_z)
    renyi_entropy = 1 / (1 - alpha) * np.log2(np.sum(np.power(normalized_z, alpha)))
    max_coff_vector = np.max(W)
    max_base_vector = np.max(H)
    # 计算标准差
    std_coff_vector = np.std(W, ddof=1)  # ddof=1 表示使用样本标准差
    std_base_vector = np.std(H, ddof=1)
    # 计算峰度
    kurt_coff_vector = pd.Series(np.squeeze(W)).kurt()
    kurt_base_vector = pd.Series(np.squeeze(H)).kurt()
    # 计算偏度
    skewness_coff_vector = pd.Series(np.squeeze(W)).skew()
    skewness_base_vector = pd.Series(np.squeeze(H)).skew()
    feature_vector = [sparsity_coff_vector, sparsity_base_vector, discontinuity_coff_vector, discontinuity_base_vector, renyi_entropy,
                      max_coff_vector, max_base_vector, std_coff_vector, std_base_vector, kurt_coff_vector, kurt_base_vector, skewness_coff_vector,
                      skewness_base_vector, kurt_time, skewness_time]
    return feature_vector


def wc_cluster():
    # feature_matrix = np.empty((0, 13), dtype=float)
    # for name in WC:
    #     feature_vector = compute_tf_feature(name, split=False)
    #     feature_matrix = np.vstack((feature_matrix, np.array(feature_vector).reshape(1, -1)))
    # print(feature_matrix)
    # # 保存为CSV格式
    # np.save('feature_matrix/feature_matrix.npy', feature_matrix)
    feature_matrix = np.load('feature_matrix/feature_matrix.npy')
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    class_number = 5
    wc_class = []
    gmm = GaussianMixture(n_components=class_number, covariance_type='full', n_init=100, tol=1e-5, reg_covar=1e-3).fit(feature_matrix)
    labels = gmm.predict(feature_matrix)
    means = gmm.means_

    for i in range(class_number):
        wc_class.append([])

    for i, label in enumerate(labels):
        wc_class[label].append(WC[i])

    for i in range(class_number):
        print(f"class {i + 1}", wc_class[i])

    with open(f'wc_class/wc_num_{class_number}.json', 'w', encoding='utf-8') as f:
        json.dump(wc_class, f, indent=4)

    tsne = TSNE(n_components=2, random_state=0)  # 修改为二维
    # 对整个特征矩阵应用t-SNE
    reduced_features = tsne.fit_transform(feature_matrix)
    # reduced_means = tsne.fit_transform(np.vstack((feature_matrix, means)))
    # reduced_means = reduced_means[-class_number:, :]

    fig = plt.figure(layout="constrained", figsize=(11, 8))
    ax = fig.add_subplot(111)  # 添加二维坐标轴
    markers = ['o', 's', '*', 'X', 'D']  # 定义两种marker
    markersize = [100, 100, 150, 100, 100]  # 设置统一的markersize
    centroidsize = [x * 2 for x in markersize]

    # 可视化聚类结果
    for label, marker in zip(np.unique(labels), markers):
        # 选择当前标签的点
        points = reduced_features[labels == label]
        ax.scatter(points[:, 0], points[:, 1], marker=marker, s=markersize[label], label=f'Cluster {label + 1}')
        centroid_x = np.mean(points[:, 0])
        centroid_y = np.mean(points[:, 1])
        ax.scatter(centroid_x, centroid_y, s=centroidsize[label], marker=marker, edgecolors='red', label=f'Centroid {label + 1}')

    # 设置图表标题和坐标轴标签
    # ax.set_title('t-SNE Visualization of GMM Clustering', fontsize=15)
    ax.set_xlabel('Feature 1', fontsize=25)
    ax.set_ylabel('Feature 2', fontsize=25)
    ax.set_xlim([-14, 15])
    # 单独设置x轴刻度字体大小
    ax.tick_params(axis='x', labelsize=15)
    # 单独设置y轴刻度字体大小
    ax.tick_params(axis='y', labelsize=15)
    # ax.legend(loc='upper left', bbox_to_anchor=(0, 0.9), ncol=1, frameon=False, fontsize=15)
    ax.legend(loc='upper left', ncol=class_number, frameon=False, fontsize=18, bbox_to_anchor=(-0.03, 1.14), columnspacing=0.5)
    ax.grid(True)
    # for i, centroid in enumerate(reduced_means):
    #     ax.scatter(centroid[0], centroid[1], s=100, c='red', label=f'Centroid {i}')
    # 显示图表
    plt.show()
    fig.savefig(f"images/tsne-{class_number}.pdf", dpi=600)


def wc_cluster_split():
    # feature_matrix = np.empty((0, 13), dtype=float)
    # temp_feature_matrix = np.empty((0, 13), dtype=float)
    # one_wc_feature_vec = None
    WCs = os.listdir('../Data/matdata_split/')
    # for name in WCs:
    #     dir = '../Data/matdata_split/' + name
    #     mat_files = glob.glob(os.path.join(dir, '*.mat'))
    #     mat_files = [s.replace("\\", "/") for s in mat_files]
    #     for file in mat_files:
    #         feature_vector = compute_tf_feature(file, split=True)
    #         temp_feature_matrix = np.vstack((temp_feature_matrix, np.array(feature_vector).reshape(1, -1)))
    #         one_wc_feature_vec = np.mean(temp_feature_matrix, axis=0)
    #     feature_matrix = np.vstack((feature_matrix, one_wc_feature_vec.reshape(1, -1)))
    #     print(f'{name} TF Feature Extraction Success!')
    # print(feature_matrix)
    # 保存为CSV格式
    # np.save('feature_matrix/feature_matrix_split.npy', feature_matrix)
    feature_matrix = np.load('feature_matrix/feature_matrix_split.npy')
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    class_number = 5
    wc_class = []
    gmm = GaussianMixture(n_components=class_number, covariance_type='full', n_init=100, tol=1e-5, reg_covar=1e-3).fit(feature_matrix)
    labels = gmm.predict(feature_matrix)
    means = gmm.means_

    for i in range(class_number):
        wc_class.append([])

    for i, label in enumerate(labels):
        wc_class[label].append(WCs[i])

    for i in range(class_number):
        print(f"class {i + 1}", wc_class[i])

    with open(f'wc_class/wc_num_{class_number}_split.json', 'w', encoding='utf-8') as f:
        json.dump(wc_class, f, indent=4)

    tsne = TSNE(n_components=2, random_state=0)  # 修改为二维
    # 对整个特征矩阵应用t-SNE
    reduced_features = tsne.fit_transform(feature_matrix)
    # reduced_means = tsne.fit_transform(np.vstack((feature_matrix, means)))
    # reduced_means = reduced_means[-class_number:, :]

    fig = plt.figure(layout="constrained", figsize=(11, 8))
    ax = fig.add_subplot(111)  # 添加二维坐标轴
    markers = ['o', 's', '*', 'X', 'D']  # 定义两种marker
    markersize = [100, 100, 150, 100, 100]  # 设置统一的markersize
    centroidsize = [x * 2 for x in markersize]

    # 可视化聚类结果
    for label, marker in zip(np.unique(labels), markers):
        # 选择当前标签的点
        points = reduced_features[labels == label]
        ax.scatter(points[:, 0], points[:, 1], marker=marker, s=markersize[label], label=f'Cluster {label + 1}')
        centroid_x = np.mean(points[:, 0])
        centroid_y = np.mean(points[:, 1])
        ax.scatter(centroid_x, centroid_y, s=centroidsize[label], marker=marker, edgecolors='red', label=f'Centroid {label + 1}')

    # 设置图表标题和坐标轴标签
    # ax.set_title('t-SNE Visualization of GMM Clustering', fontsize=15)
    ax.set_xlabel('Feature 1', fontsize=25)
    ax.set_ylabel('Feature 2', fontsize=25)
    # ax.set_xlim([-14, 15])
    # 单独设置x轴刻度字体大小
    ax.tick_params(axis='x', labelsize=15)
    # 单独设置y轴刻度字体大小
    ax.tick_params(axis='y', labelsize=15)
    # ax.legend(loc='upper left', bbox_to_anchor=(0, 0.9), ncol=1, frameon=False, fontsize=15)
    ax.legend(loc='upper left', ncol=class_number, frameon=False, fontsize=18, bbox_to_anchor=(-0.03, 1.14), columnspacing=0.5)
    ax.grid(True)
    # for i, centroid in enumerate(reduced_means):
    #     ax.scatter(centroid[0], centroid[1], s=100, c='red', label=f'Centroid {i}')
    # 显示图表
    plt.show()
    fig.savefig(f"images/tsne-{class_number}.pdf", dpi=600)


def three_d_tsne(feature_matrix, labels):
    # 初始化t-SNE
    tsne = TSNE(n_components=3, random_state=0)
    # 对整个特征矩阵应用t-SNE
    reduced_features = tsne.fit_transform(feature_matrix)

    fig = plt.figure(layout="constrained", figsize=(10, 7))
    elevation = 25
    azimuth = -132
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)  # 使用 elev 和 azim 参数设置视角
    markers = ['o', 's', '*']  # 定义两种marker
    markersize = 50  # 设置统一的markersize

    # 可视化聚类结果
    for label, marker in zip(np.unique(labels), markers):
        # 选择当前标签的点
        points = reduced_features[labels == label]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker=marker, s=markersize, label=f'Cluster {label + 1}')
    # # 可视化聚类中心
    # for i, centroid in enumerate(means):
    #     ax.scatter(centroid[0], centroid[1], centroid[2], s=100, c='red', label=f'Centroid {i}')

    # 设置图表标题和坐标轴标签
    # ax.set_title('t-SNE Visualization of GMM Clustering')
    ax.set_xlabel('Feature 1', fontsize=15)
    ax.set_ylabel('Feature 2', fontsize=15)
    ax.set_zlabel('Feature 3', fontsize=15)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.9), ncol=1, frameon=False, fontsize=15)


def feature_evaluation(X, y):
    """
    X: 特征矩阵 (样本数 × 特征数)
    y: 工况标签
    """
    n_features = X.shape[1]
    results = {}

    # 1. ANOVA分析
    f_values = []
    p_values = []
    for i in range(n_features):
        f_val, p_val = stats.f_oneway(*[X[y == c, i] for c in np.unique(y)])
        f_values.append(f_val)
        p_values.append(p_val)

    # 2. Fisher判别准则
    fisher_scores = []
    for i in range(n_features):
        # 计算类间距离
        class_means = [np.mean(X[y == c, i]) for c in np.unique(y)]
        between_class_var = np.var(class_means)

        # 计算类内距离
        within_class_var = np.mean([np.var(X[y == c, i]) for c in np.unique(y)])

        # Fisher score
        fisher_score = between_class_var / (within_class_var + 1e-6)
        fisher_scores.append(fisher_score)

    # 3. 互信息
    mi_scores = mutual_info_classif(X, y)

    results['f_values'] = f_values
    results['p_values'] = p_values
    results['fisher_scores'] = fisher_scores
    results['mi_scores'] = mi_scores

    return results


def print_feature_importance(results, feature_names=None):
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(results['f_values']))]

    # 1. 基于F值排序
    f_indices = np.argsort(results['f_values'])[::-1]
    print("\nFeature Importance Ranking (Based on F-value):")
    print("Rank  Feature Name          F-value      p-value")
    print("-" * 55)
    for rank, idx in enumerate(f_indices, 1):
        print(f"{rank:2d}.   {feature_names[idx]:<20} {results['f_values'][idx]:10.2f} {results['p_values'][idx]:10.4f}")

    # 2. 基于Fisher分数排序
    fisher_indices = np.argsort(results['fisher_scores'])[::-1]
    print("\nFeature Importance Ranking (Based on Fisher Criterion):")
    print("Rank  Feature Name          Fisher Score")
    print("-" * 45)
    for rank, idx in enumerate(fisher_indices, 1):
        print(f"{rank:2d}.   {feature_names[idx]:<20} {results['fisher_scores'][idx]:10.2f}")

    # 3. 基于互信息排序
    mi_indices = np.argsort(results['mi_scores'])[::-1]
    print("\nFeature Importance Ranking (Based on Mutual Information):")
    print("Rank  Feature Name          MI Score")
    print("-" * 45)
    for rank, idx in enumerate(mi_indices, 1):
        print(f"{rank:2d}.   {feature_names[idx]:<20} {results['mi_scores'][idx]:10.2f}")

    # 综合评分
    ranks = np.zeros(len(feature_names))
    for i, idx in enumerate(f_indices):
        ranks[idx] += i
    for i, idx in enumerate(fisher_indices):
        ranks[idx] += i
    for i, idx in enumerate(mi_indices):
        ranks[idx] += i

    comprehensive_indices = np.argsort(ranks)
    print("\nFeature Importance Ranking (Comprehensive Score):")
    print("Rank  Feature Name          Combined Rank")
    print("-" * 45)
    for rank, idx in enumerate(comprehensive_indices, 1):
        print(f"{rank:2d}.   {feature_names[idx]:<20} {ranks[idx]:10.2f}")


def plot_feature(WC):
    # plt.style.use('seaborn-v0_8-paper')
    markers = ["^", "^", "^", "^", "v", "v", "v", "v"]
    length = 15
    feature_matrix = np.empty((0, length), dtype=float)
    for name in WC:
        feature_vector = compute_tf_feature(name, split=False)
        feature_matrix = np.vstack((feature_matrix, np.array(feature_vector).reshape(1, -1)))
        # 绘制特征向量
    feature_names = [
        'Sparsity_Coeff', 'Sparsity_Base', 'Discontinuity_Coeff', 'Discontinuity_Base', 'Renyi_Entropy',
        'Max_Coeff', 'Max_Base', 'Std_Coeff', 'Std_Base', 'Kurt_Coeff', 'Kurt_Base',
        'Skewness_Coeff', 'Skewness_Base', 'Kurt_Time', 'Skewness_Time'
    ]
    y = np.zeros((8, 1))
    y[4:] = 1
    results = feature_evaluation(feature_matrix, np.squeeze(y))
    print_feature_importance(results, feature_names=feature_names)
    plot_feature_distributions(feature_matrix, np.squeeze(y), feature_names=feature_names)
    plot_feature_distributions_single(feature_matrix, feature_names)
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for i in range(feature_matrix.shape[0]):
        plt.plot(feature_matrix[i], label=f'Feature vector {i + 1}', linewidth=2, marker=markers[i], markersize=10)
    plt.xlabel('Feature Index', fontsize=15)
    plt.ylabel('Feature Value', fontsize=15)
    plt.legend(WC, ncol=2, frameon=False, fontsize=15)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=13)
    ax.set_xticks(ticks=np.arange(0, length), labels=np.arange(1, length + 1))
    plt.show()
    fig.savefig('./images/feature-indicator-RE-RP.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # # 设置默认字号
    # mpl.rcParams['font.size'] = 18
    # 设置默认字体，这里以'SimHei'为例，表示黑体
    mpl.rcParams['font.family'] = 'Times New Roman'

    # wc_cluster()
    # figure_stft()
    # segment_figure()
    # segment_fusion_figure()
    # wc_cluster_split()
    WC1 = ["BC30_01", "BC30_02", "BC40_01", "BC40_02", "BR30_01", "BR30_02", "BR40_01", "BR40_02"]
    WC2 = ["RE30_01", "RE30_02", "RE40_01", "RE40_02", "RP30_01", "RP30_02", "RP40_01", "RP40_02"]
    WC3 = ["BK30_01", "BK30_02", "BK50_01", "BK50_02", "CJ40_01", "CJ40_02", "CJ50_01", "CJ50_02"]
    plot_feature(WC2)
