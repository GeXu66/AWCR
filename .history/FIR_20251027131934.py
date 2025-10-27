import numpy as np
import torch
from matplotlib import rcParams
import scipy.io as scio
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import signal


def is_valid_tuple(input_data):
    if isinstance(input_data, tuple) and len(input_data) == 2:
        start, end = input_data
        if isinstance(start, int) and isinstance(end, int) and start > 0 and end > 0 and start < end:
            return True
    return False


def ridge_reg(K, out):
    cond = np.linalg.cond(K)
    print('condition number', cond)
    gama = K.T @ K
    C_lim = 1e9
    eigvals = np.linalg.eigvals(gama)
    # 获取A的最大和最小特征值
    max_eigval = np.real(np.max(eigvals))
    min_eigval = np.real(np.min(eigvals))
    if cond > C_lim:
        lam = (max_eigval - min_eigval * C_lim) / (C_lim - 1)
    else:
        lam = 0
    inv_K = np.linalg.inv(gama + lam * np.eye(gama.shape[0]))
    alpha = inv_K @ K.T @ out
    print('lamda:', lam)
    return alpha


# 添加噪声生成函数
def generate_pink_noise(size, gamma=1.0):
    """
    生成粉红噪声 (Pink Noise)
    PSD ∝ 1/f^γ
    
    参数:
    size: 噪声长度
    gamma: 频谱密度系数，通常为1.0
    """
    if isinstance(size, int):
        size = (size, 1)
    
    # 生成白噪声
    white_noise = np.random.normal(0, 1, size)
    
    # 应用傅里叶变换
    if size[1] == 1:
        # 单通道情况
        X = np.fft.rfft(white_noise[:, 0])
        S = np.arange(1, len(X) + 1)
        S = np.sqrt(1.0 / np.power(S, gamma/2.0))
        y = np.real(np.fft.irfft(X * S))
        return y.reshape(-1, 1)
    else:
        # 多通道情况
        result = np.zeros(size)
        for i in range(size[1]):
            X = np.fft.rfft(white_noise[:, i])
            S = np.arange(1, len(X) + 1)
            S = np.sqrt(1.0 / np.power(S, gamma/2.0))
            result[:, i] = np.real(np.fft.irfft(X * S))
        return result


def generate_impulsive_noise(size, epsilon=0.1, sigma_b=1.0, sigma_i=10.0):
    """
    生成脉冲噪声 (Impulsive Noise)
    n(t) ~ (1-ε)N(0,σ_b²) + εN(0,σ_i²)
    
    参数:
    size: 噪声大小
    epsilon: 脉冲概率
    sigma_b: 背景噪声标准差
    sigma_i: 脉冲噪声标准差
    """
    if isinstance(size, int):
        size = (size, 1)
    
    # 生成随机选择矩阵，决定哪些点是脉冲
    mask = np.random.random(size) < epsilon
    
    # 生成背景噪声
    background_noise = np.random.normal(0, sigma_b, size)
    
    # 生成脉冲噪声
    impulse_noise = np.random.normal(0, sigma_i, size)
    
    # 合并噪声
    noise = background_noise * (1 - mask) + impulse_noise * mask
    
    return noise


def generate_cross_channel_noise(size, num_channels, correlation_strength=0.5):
    """
    生成跨通道相关噪声 (Cross-Channel Noise)
    n(t) ~ N(0, Σ)
    
    参数:
    size: 每个通道的样本数
    num_channels: 通道数
    correlation_strength: 通道间相关强度 (0~1)
    """
    # 创建相关矩阵
    corr_matrix = np.eye(num_channels)
    for i in range(num_channels):
        for j in range(num_channels):
            if i != j:
                corr_matrix[i, j] = correlation_strength
    
    # 确保协方差矩阵是正定的
    min_eig = np.min(np.real(np.linalg.eigvals(corr_matrix)))
    if min_eig < 0:
        corr_matrix += np.eye(num_channels) * (abs(min_eig) + 0.01)
    
    # 生成多维高斯噪声
    noise = np.random.multivariate_normal(np.zeros(num_channels), corr_matrix, size)
    
    return noise


class FIR:
    def __init__(self, model_order, channel_in, channel_out, mean_flag, trainName):
        self.model_order = model_order
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.phi = None
        self.theta = None
        self.ni = len(channel_in)
        self.no = len(channel_out)
        self.mean_flag = mean_flag
        self.input_length = self.model_order * self.ni
        if isinstance(trainName, list):
            self.trainfile = ['../Data/matdata/' + n + '.mat' for n in trainName]
        else:
            self.trainfile = ['../Data/matdata/' + trainName + '.mat']

    def read_mat(self, piece, noise_channel=None, noise_type=None, noise_params=None):
        train = []
        target = []
        for file in self.trainfile:
            key = file.split('/')[-1]
            key = key.split('.')[-2]
            data = scio.loadmat(file)
            data = data[key]

            # 应用噪声
            if noise_type is not None:
                # 单通道噪声（旧功能保留）
                if noise_channel is not None and isinstance(noise_channel, int):
                    if noise_type == 'bias':
                        length = data[:, noise_channel].shape[0]
                        part1 = int(length * 0.2)
                        part2 = int(length * 0.4)
                        data[:part1, noise_channel] += 1  # 前20%为随机+1
                        data[part1:part1 + part2, noise_channel] -= 3  # 中间40%为随机-3
                        data[part1 + part2:, noise_channel] -= 1  # 后40%为随机-1
                    elif noise_type == 'gaussian':
                        data[:, noise_channel] += np.random.normal(0, 1, data[:, noise_channel].shape)  # 添加高斯噪声
                    elif noise_type == 'shrink':
                        data[:, noise_channel] *= 0.8  # 对指定通道整体乘以0.8
                
                # 多通道噪声（新功能）
                elif noise_channel is not None and isinstance(noise_channel, list):
                    sample_length = data.shape[0]
                    num_channels = len(noise_channel)
                    
                    if noise_type == 'pink':
                        # 应用粉红噪声
                        gamma = 1.0 if noise_params is None or 'gamma' not in noise_params else noise_params['gamma']
                        scale = 0.1 if noise_params is None or 'scale' not in noise_params else noise_params['scale']
                        noise = generate_pink_noise((sample_length, num_channels), gamma) * scale
                        for i, ch in enumerate(noise_channel):
                            data[:, ch] += noise[:, i]
                    
                    elif noise_type == 'impulsive':
                        # 应用脉冲噪声
                        epsilon = 0.1 if noise_params is None or 'epsilon' not in noise_params else noise_params['epsilon']
                        sigma_b = 0.1 if noise_params is None or 'sigma_b' not in noise_params else noise_params['sigma_b']
                        sigma_i = 1.0 if noise_params is None or 'sigma_i' not in noise_params else noise_params['sigma_i']
                        noise = generate_impulsive_noise((sample_length, num_channels), epsilon, sigma_b, sigma_i)
                        for i, ch in enumerate(noise_channel):
                            data[:, ch] += noise[:, i]
                    
                    elif noise_type == 'cross_channel':
                        # 应用跨通道相关噪声
                        corr_strength = 0.5 if noise_params is None or 'corr_strength' not in noise_params else noise_params['corr_strength']
                        scale = 0.1 if noise_params is None or 'scale' not in noise_params else noise_params['scale']
                        noise = generate_cross_channel_noise(sample_length, num_channels, corr_strength) * scale
                        for i, ch in enumerate(noise_channel):
                            data[:, ch] += noise[:, i]

            train.append(data[:, self.channel_in])
            target.append(data[:, self.channel_out])

            print('-------------------Loading Data-------------------')
            print(f'data {key} load successfully!')

        if len(train) > 1:
            train_all = train[0]
            target_all = target[0]
            for i in range(1, len(train)):
                train_all = np.vstack((train_all, train[i]))
                target_all = np.vstack((target_all, target[i]))
        else:
            train_all = train[0]
            target_all = target[0]

        if self.mean_flag:
            train_all = (train_all - np.mean(train_all, axis=0))  # / np.std(train_all, axis=0)
            target_all = (target_all - np.mean(target_all, axis=0))  # / np.std(target_all, axis=0)

        if is_valid_tuple(piece) and piece[1] <= train_all.shape[0]:
            self.time_length = piece[1] - piece[0] + 1
            self.train_all = train_all[piece[0]:piece[1] + 1, :]
            self.target_all = target_all[piece[0]:piece[1] + 1, :]
        elif piece == 'all':
            self.time_length = train_all.shape[0]
            self.train_all = train_all
            self.target_all = target_all
        else:
            print('input length wrong!')
            exit(0)

    def gen_phi(self):
        phi = np.zeros((self.train_all.shape[0], self.input_length))
        new_row = np.tile(self.train_all[0], (self.model_order - 1, 1))
        train_all = np.vstack((new_row, self.train_all))
        for i in range(self.time_length):
            phi[i, :] = train_all[i:i + self.model_order, :].T.reshape(1, -1)
        self.phi = phi

    def gen_phi_data(self, data, order):
        phi_data = np.zeros((data.shape[0], data.shape[1] * order))
        new_row = np.tile(data[0], (order - 1, 1))
        slice_data_all = np.vstack((new_row, data))
        for i in range(data.shape[0]):
            phi_data[i, :] = slice_data_all[i:i + order, :].T.reshape(1, -1)
        return phi_data

    def train(self):
        print('-------------------Training-------------------')
        self.theta = ridge_reg(self.phi, self.target_all)
        print(self.theta)
        return self.theta

    def predict(self, u):
        predict_scale = u.shape
        print('test input shape', predict_scale)
        phi_u = self.gen_phi_data(data=u, order=self.model_order)
        print('test input recurrent shape', phi_u.shape)
        print('-------------------Predicting-------------------')
        predict_out = phi_u @ self.theta
        return predict_out


def fir_prediction(model_order, channel_in, channel_out, mean_flag, trainName, testName, data_volume, noise_channel, noise_type, noise_params=None):
    fir = FIR(model_order, channel_in, channel_out, mean_flag, trainName)
    test_fir = FIR(model_order, channel_in, channel_out, mean_flag, testName)
    fir.read_mat(data_volume, None, None)
    test_fir.read_mat(data_volume, noise_channel, noise_type, noise_params)
    fir.gen_phi()
    fir.train()
    predict_out = fir.predict(test_fir.train_all)
    real_out = test_fir.target_all
    # plot_outputs(predict_out, real_out)
    return real_out, predict_out


def plot_outputs(predict_out, real_out):
    # Flatten the arrays if they are not already 1-dimensional
    predict_out = np.ravel(predict_out)
    real_out = np.ravel(real_out)

    # Plotting
    plt.figure(figsize=(10, 5))  # Set the figure size as desired
    plt.plot(real_out, label='Real Output', color='blue', linewidth=2)
    plt.plot(predict_out, label='Predicted Output', color='orange', linewidth=2)

    # Make the plot aesthetic
    # plt.title('Predicted vs Real Output')  # Set the title
    plt.xlabel('Sample Index', fontsize=15)  # Set the x-axis label
    plt.ylabel('Output Value', fontsize=15)  # Set the y-axis label
    plt.legend()  # Show the legend
    plt.grid(True)  # Enable the grid

    # Optionally, you can set the limits for the axes if you want to zoom in on a particular range
    # plt.xlim(xmin, xmax)
    # plt.ylim(ymin, ymax)

    # Show the plot
    plt.show()


def compute_metric(real_out, predict_out):
    rmse = np.sqrt(mean_squared_error(real_out, predict_out))
    mae = mean_absolute_error(real_out, predict_out)
    r2 = r2_score(real_out, predict_out)
    return rmse, mae, r2


def evaluate_noise_robustness(model_order, channel_in, channel_out, mean_flag, trainName, testName, data_volume, 
                             output_dir='./noise_robustness_results'):
    """
    评估模型在不同噪声条件下的鲁棒性
    
    参数:
    model_order: 模型阶数
    channel_in: 输入通道列表
    channel_out: 输出通道列表
    mean_flag: 是否去均值标志
    trainName: 训练数据名称
    testName: 测试数据名称
    data_volume: 数据量
    output_dir: 结果输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 首先获取无噪声情况下的真实输出
    fir = FIR(model_order, channel_in, channel_out, mean_flag, trainName)
    test_fir = FIR(model_order, channel_in, channel_out, mean_flag, testName)
    fir.read_mat(data_volume)
    test_fir.read_mat(data_volume)
    fir.gen_phi()
    fir.train()
    real_out = test_fir.target_all
    
    # 定义要测试的噪声类型和参数
    noise_types = [
        {
            'name': 'pink',
            'params': {'gamma': 1.0, 'scale': 0.05},
            'description': 'Pink Noise (1/f)'
        },
        {
            'name': 'impulsive',
            'params': {'epsilon': 0.05, 'sigma_b': 0.05, 'sigma_i': 1.0},
            'description': 'Impulsive Noise (Low)'
        },
        {
            'name': 'cross_channel',
            'params': {'corr_strength': 0.3, 'scale': 0.05},
            'description': 'Cross-Channel Noise (Low Correlation)'
        },
    ]
    
    # 创建结果DataFrame
    results = pd.DataFrame()
    results['Time'] = np.arange(len(real_out))
    results['Actual'] = real_out
    
    # 对每种噪声类型进行测试
    for noise_type in noise_types:
        print(f"Testing with {noise_type['description']}...")
        
        # 应用噪声并预测
        _, predict_out = fir_prediction(
            model_order, channel_in, channel_out, mean_flag,
            trainName, testName, data_volume, 
            channel_in,  # 对所有输入通道应用噪声
            noise_type['name'], 
            noise_type['params']
        )
        
        # 添加到结果DataFrame
        results[noise_type['description']] = predict_out
        
        # 计算评估指标
        rmse, mae, r2 = compute_metric(real_out, predict_out)
        print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")
    
    # 保存结果到CSV
    testName = testName[0]
    csv_file = os.path.join(output_dir, f"{testName}_noise_robustness.csv")
    results.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")
    
    return results


def class_error(train, test, data_volume):
    noise_channel = [list(range(90, 106))]
    noise_channel = [x for sublist in noise_channel for x in sublist]
    # noise_channel = 111
    noise_type = 'bias'
    model_order = 1
    # old tire
    real_out_old, predict_out_old = fir_prediction(model_order, tradition_channel_in, channel_out, mean_flag, train, test, data_volume, None, noise_type)

    rmse_old = mean_absolute_percentage_error(real_out_old, predict_out_old)
    rmse_old = np.sqrt(rmse_old) * 2
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    # 配置样式
    x = np.arange(1, len(real_out_old) + 1)
    x = x / 500
    # 绘制曲线
    color1, color2, color3 = '#f6541f', '#2299f0', '#72b607'
    ax.plot(x, real_out_old, linestyle='-', label='Measured', color=color1, linewidth=3)
    ax.plot(x, predict_out_old, linestyle='--', label='Estimated', color=color3, linewidth=3)
    ax.text(0.05, 0.05, f'RMSE: {rmse_old:.2f}%', transform=ax.transAxes, fontsize=14, color='black', ha='left', va='bottom')
    # 设置图例、轴标签、刻度等
    label_size = 22
    legend_size = 16
    tick_size = 16
    ax.set_xlabel('Time (s)', fontsize=label_size)
    ax.set_ylabel('Vertical Force', fontsize=label_size)
    ax.legend(frameon=True, edgecolor='black', fontsize=legend_size, fancybox=False, ncol=1, loc='upper left')
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=tick_size)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    # plt.grid(True, axis='y', linestyle='--', linewidth=1)
    plt.show()
    # 显示图像并保存
    fig.savefig(f'./images/compare_class_error_{test}.pdf', dpi=600)
    
    # 保存数据到CSV文件
    output_dir = './class_error_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建DataFrame保存结果
    results = pd.DataFrame()
    results['Time'] = x
    results['Measured'] = np.ravel(real_out_old)
    results['Estimated'] = np.ravel(predict_out_old)
    results['RMSE'] = rmse_old
    
    # 保存结果到CSV
    filename = f"{train}_to_{test}"
    if isinstance(data_volume, tuple):
        filename += f"_{data_volume[0]}_{data_volume[1]}"
    elif data_volume != 'all':
        filename += f"_{data_volume}"
    
    csv_file = os.path.join(output_dir, f"{filename}.csv")
    results.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    rcParams['mathtext.default'] = 'regular'
    tradition_channel_in = [list(range(90, 106)), list(range(109, 112)), list(range(118, 121))]
    tradition_channel_in = [x for sublist in tradition_channel_in for x in sublist]
    channel_in = [list(range(0, 32)), list(range(56, 139))]
    channel_in = [x for sublist in channel_in for x in sublist]
    remove_list = [57, 65, 76, 79]
    channel_in = [x for x in channel_in if x not in remove_list]
    # channel_in = [69, 78, 83, 84, 91, 94, 95, 96]
    channel_out = []
    channel_out = [34]
    model_order = 50
    mean_flag = True
    
    trainlist = [
        ['SW20_01', 'SW20_02'], ['SW25_01', 'SW25_02'], ['SW35_01', 'SW35_02'], ['RE20_01', 'RE20_02'], ['RE30_01', 'RE30_02'], ['RE40_01', 'RE40_02'],
        ['CJ30_01', 'CJ30_02'], ['CJ40_01', 'CJ40_02'], ['CJ50_01', 'CJ50_02'],
    ]
    testlist = [
        ['SW20_03'], ['SW25_03'], ['SW35_03'], ['RE20_03'], ['RE30_03'], ['RE40_03'],
        ['CJ30_03'], ['CJ40_03'], ['CJ50_03'],
    ]
    # for train, test in zip(trainlist, testlist):
    #     evaluate_noise_robustness(model_order, tradition_channel_in, channel_out, mean_flag, train, test, 'all')
    # class_error(train='ST25_01', test='TW25_01', data_volume=(1000, 4000))
    # class_error(train='BR40_01', test='RE40_01', data_volume=(1500, 4500))
    class_error(train='BR30_01', test='SW35_01', data_volume='all')
    class_error(train='PH40_01', test='DP30_01', data_volume='all')