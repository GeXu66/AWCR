import os
import joblib
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import *
import matplotlib.pyplot as plt
import seaborn as sns

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, model_type, bi_dir, out_channel_num):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 2
        self.model_type = model_type
        self.device = torch.device(device)
        # self.checkpoint_file = "model/1-env-mul-vel/LSTM"
        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=bi_dir
        )
        self.gru = nn.GRU(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=bi_dir
        )
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=out_channel_num)

    def forward(self, x):
        # batch_size = x.shape[0]
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(device)
        # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(device)
        if self.model_type == 'LSTM':
            _, (hn, _) = self.lstm(x)
            hn = hn.to(self.device)
            out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        else:
            _, hn = self.gru(x)
            hn = hn.to(self.device)
            out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        return out

    def save_checkpoint(self, name):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), name)

    def load_checkpoint(self, name):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(name))


class SequenceDataset(Dataset):
    def __init__(self, input, target, sequence_length=10):
        self.target = target
        self.sequence_length = sequence_length
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.y = torch.tensor(target).float().to(self.device)
        self.X = torch.tensor(input).float().to(self.device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[i]


def read_mat(filename, lstm_class, save, load, feature=30):
    train = []
    target = []
    for file in filename:
        key = file.split('/')[-1]
        key = key.split('.')[-2]
        data = scio.loadmat(file)
        data = data[key]
        # train.append(np.hstack((data[:, 90:106], data[:, 109:112], data[:, 118:121])))
        train.append(np.hstack((data[:, 4:8], data[:, 56:138])))
        target.append(np.reshape(data[:, 34], (-1, 1)))
        print(f'data {key} load successfully!')
    train_all = train[0]
    target_all = target[0]
    for i in range(1, len(train)):
        train_all = np.vstack((train_all, train[i]))
        target_all = np.vstack((target_all, target[i]))
    train_all, target_all, all_scaler = data_transform(train_all=train_all, target=target_all, feature=feature, save=save, load=load, lstm_class=lstm_class)
    return train_all, target_all, all_scaler


def load_scaler(lstm_class):
    save_scaler_dir = f'scaler/class_{lstm_class}/'
    scaler_filename = save_scaler_dir + f"predict_scaler_class_{lstm_class}.save"
    predict_scaler = joblib.load(scaler_filename)
    print('load predict scaler successful!')
    scaler_filename = save_scaler_dir + f"pca_scaler_class_{lstm_class}.save"
    pca_scaler = joblib.load(scaler_filename)
    print('load pca scaler successful!')
    scaler_filename = save_scaler_dir + f"std_scaler_class_{lstm_class}.save"
    std_scaler = joblib.load(scaler_filename)
    print('load std scaler successful!')
    return predict_scaler, pca_scaler, std_scaler


def data_transform(train_all, target, feature, save, load, lstm_class):
    if load:
        predict_scaler, pca_scaler, std_scaler = load_scaler(lstm_class)
        target = target.reshape(-1, 1)
        target_trans = predict_scaler.transform(target)
        target_trans = np.squeeze(target_trans)
        # transform train data
        train_all_std = std_scaler.transform(train_all)
        pca_train_all = pca_scaler.transform(train_all_std)
        # print(pca_scaler.explained_variance_ratio_)
    else:
        # STD scaler
        std_scaler = StandardScaler()
        # PCA scaler
        pca_scaler = PCA(n_components=feature)
        # predict scaler
        predict_scaler = MinMaxScaler(feature_range=(-5000, 5000))
        # predict_scaler = StandardScaler()
        target = target.reshape(-1, 1)
        predict_scaler.fit(target)
        # transform target data
        # print('Min: %f, Max: %f' % (predict_scaler.data_min_, predict_scaler.data_max_))
        target_trans = predict_scaler.transform(target)
        target_trans = np.squeeze(target_trans)
        # transform train data
        std_scaler.fit(train_all)
        train_all_std = std_scaler.transform(train_all)
        pca_scaler.fit(train_all_std)
        pca_train_all = pca_scaler.transform(train_all_std)
        print(pca_scaler.explained_variance_ratio_)
    all_scaler = {'predict_scaler': predict_scaler, 'pca_scaler': pca_scaler, 'std_scaler': std_scaler}
    if save:
        save_scaler_dir = f'scaler/class_{lstm_class}/'
        if not os.path.exists(save_scaler_dir):  # 判断所在目录下是否有该文件名的文件夹
            os.makedirs(save_scaler_dir)  # 创建多级目录用mkdirs，单击目录mkdir
        scaler_filename = save_scaler_dir + f"predict_scaler_class_{lstm_class}.save"
        joblib.dump(predict_scaler, scaler_filename)
        print(f'save predict scaler class {lstm_class} successful!')
        scaler_filename = save_scaler_dir + f"pca_scaler_class_{lstm_class}.save"
        joblib.dump(pca_scaler, scaler_filename)
        print(f'save PCA scaler class {lstm_class} successful!')
        scaler_filename = save_scaler_dir + f"std_scaler_class_{lstm_class}.save"
        joblib.dump(std_scaler, scaler_filename)
        print(f'save STD scaler class {lstm_class} successful!')
    return pca_train_all, target_trans, all_scaler


def data_transform_cluster(train_all, feature):
    # STD scaler
    std_scaler = StandardScaler()
    # PCA scaler
    kpca_scaler = KernelPCA(n_components=feature, kernel='rbf')
    # transform target data
    # print('Min: %f, Max: %f' % (predict_scaler.data_min_, predict_scaler.data_max_))
    # transform train data
    train_all_std = std_scaler.fit_transform(train_all)
    pca_train_all = kpca_scaler.fit_transform(train_all_std)
    return pca_train_all


def train_model(data_loader, model, loss_function, ix_epoch, optimizer, model_name, epoch_all):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    loop = tqdm(data_loader, total=len(data_loader))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    for X, y in loop:
        output = model(X).to(device)
        loss = loss_function(output, y).to(device)
        optimizer.zero_grad()
        loss.backward()
        # for name, parms in model.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)
        optimizer.step()
        total_loss += loss.item()
        loop.set_description(f'Epoch [{ix_epoch}/{epoch_all}]')
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches
    print(f"{model_name} Train loss: {avg_loss}")
    return avg_loss


def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            total_loss += loss_function(output, y).item()
    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")


def predict(data_loader, model):
    output = torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            X = X.to(device)
            y_star = model(X)
            y_star = y_star.to(device)
            output = torch.cat((output, y_star), 0)

    return output


def draw_model(root, data_loader, model, predict_scaler, target, model_name):
    wcl_trans = predict(data_loader, model).cpu().numpy()
    wcl_trans = wcl_trans.reshape(-1, 1)
    # wcl = predict_scaler.inverse_transform(wcl_trans)
    wcl = np.squeeze(wcl_trans)
    print('length:', len(wcl))
    # plt.plot(target[1000:3000])
    # plt.plot(wcl[1000:3000])
    # plt.show()
    scio.savemat(f'{root}/{model_name}_predict.mat', {f'{model_name}_predict': wcl.reshape(-1, 1)})
    scio.savemat(f'{root}/wcl_real.mat', {f'wcl_real': target.reshape(-1, 1)})


def validation(model, model_name, testfile, channel, sequence_length, dir_name, feature, lstm_class):
    batch_size = 1024
    for file in testfile:
        raw_file = file.split('/')[-1].split('.')[-2]
        print('testfile:{}'.format(raw_file))
        sub_dir = dir_name + raw_file + '/' + channel
        if not os.path.exists(sub_dir):  # 判断所在目录下是否有该文件名的文件夹
            os.makedirs(sub_dir)  # 创建多级目录用mkdirs，单击目录mkdir
        test, target_test, all_scaler = read_mat([file], lstm_class=lstm_class, save=False, load=True, feature=feature)
        # only load train scaler
        test_dataset = SequenceDataset(
            input=test,
            target=target_test,
            sequence_length=sequence_length
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        draw_model(root=sub_dir, data_loader=test_loader, model=model, predict_scaler=all_scaler['predict_scaler'],
                   target=target_test, model_name=model_name)


def plot_feature_distributions(X, y, feature_names):
    """
    绘制每个特征在不同工况下的分布箱线图

    参数:
    X: 特征矩阵 (样本数 × 特征数)
    y: 工况标签
    feature_names: 特征名称列表
    """
    n_features = X.shape[1]
    n_conditions = len(np.unique(y))

    # 设置图形风格
    plt.style.use('seaborn-v0_8-paper')

    # 计算适当的子图网格大小
    n_cols = 4  # 每行4个子图
    n_rows = int(np.ceil(n_features / n_cols))

    # 创建图形
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    fig.suptitle('Feature Distributions Across Different Conditions', fontsize=16, y=1.02)

    # 创建每个特征的箱线图
    for i in range(n_features):
        plt.subplot(n_rows, n_cols, i + 1)

        # 准备数据
        data = []
        labels = []
        for condition in np.unique(y):
            data.append(X[y == condition, i])
            labels.append(f'C{condition}')

        # 绘制箱线图
        box = plt.boxplot(data, labels=labels, patch_artist=True)

        # 设置颜色
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # 设置标题和标签
        plt.title(feature_names[i], fontsize=10)
        plt.xlabel('Condition')
        plt.ylabel('Value')

        # 调整y轴范围，确保异常值也能显示
        if len(data) > 0:
            all_values = np.concatenate(data)
            q1 = np.percentile(all_values, 25)
            q3 = np.percentile(all_values, 75)
            iqr = q3 - q1
            plt.ylim(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        # 网格线
        plt.grid(True, linestyle='--', alpha=0.7)

        # 旋转x轴标签
        plt.xticks(rotation=45)

    plt.show()

    return fig


def plot_selected_features_distributions(X, y, feature_names, selected_features):
    """
    只绘制选定特征的分布箱线图

    参数:
    X: 特征矩阵
    y: 工况标签
    feature_names: 特征名称列表
    selected_features: 选定特征的索引列表
    """
    X_selected = X[:, selected_features]
    feature_names_selected = [feature_names[i] for i in selected_features]
    return plot_feature_distributions(X_selected, y, feature_names_selected)


def plot_feature_distributions_single(X, feature_names):
    """
    在一张图上绘制所有特征的分布箱线图

    参数:
    X: 特征矩阵 (样本数 × 特征数)
    feature_names: 特征名称列表
    """
    # 设置风格
    plt.style.use('seaborn-v0_8-paper')
    # sns.set_palette("husl")

    # 创建图形
    fig, ax = plt.subplots(figsize=(6.5, 4))

    # 绘制箱线图
    bp = ax.boxplot([X[:, i] for i in range(X.shape[1])],
                    patch_artist=True,  # 填充箱体
                    medianprops=dict(color="black", linewidth=1.5),  # 中位数线
                    flierprops=dict(marker='o', markerfacecolor='gray',
                                    markersize=4, alpha=0.5),  # 异常值
                    whiskerprops=dict(linewidth=1.5),  # 须线
                    capprops=dict(linewidth=1.5))  # 须线端点

    # 设置箱体颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 设置标题和标签
    # plt.title('Distribution of Features', pad=20, fontsize=14)
    plt.xlabel('Feature Index', fontsize=15)
    plt.ylabel('Feature Value', fontsize=15)

    # # 调整x轴标签
    # plt.xticks(rotation=45, ha='right')

    # 添加网格线
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 美化坐标轴
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=13)
    ax.set_xticks(ticks=np.arange(1, len(feature_names)+1), labels=np.arange(1, len(feature_names) + 1))

    # 添加背景色
    # ax.set_facecolor('#f8f9fa')
    fig.savefig('./images/featurebox.png', dpi=300, bbox_inches='tight')

    return fig
