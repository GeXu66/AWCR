import os
import glob
from scipy.io import loadmat, savemat

if __name__ == '__main__':
    # 设置文件夹路径，这里需要替换为实际的文件夹路径
    folder_path = r'../Data/matdata'
    Tw = 5 * 512
    seg_num = 10
    # 使用glob模块获取所有.mat文件
    mat_files = glob.glob(os.path.join(folder_path, '*.mat'))

    # 初始化一个字典来存储具有相同前缀的文件名
    file_groups = {}

    # 遍历所有.mat文件
    for mat_file in mat_files:
        # 获取不包含下划线和序号的部分作为前缀
        prefix = mat_file.split(os.path.sep)[-1].split('_')[0]

        # 检查前缀是否已经在字典中
        if prefix not in file_groups:
            file_groups[prefix] = []

        # 将文件名添加到对应前缀的列表中
        file_groups[prefix].append(mat_file)

    # 遍历字典中的每个前缀和对应的文件名列表
    for prefix, files in file_groups.items():
        # 创建一个字典来存储同一前缀下的所有.mat文件的数据
        mat_data = {}
        count = 0
        # 读取同一前缀下的所有.mat文件
        for file in files:
            data = loadmat(file)
            key = file.split(os.path.sep)[-1].split('.')[0]
            data = data[key]
            len = data.shape[0]
            if len < Tw:
                pass
            else:
                dir_name = rf'../Data/matdata_split/{prefix}/'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                overlap = int(Tw - (len - Tw) / (seg_num - 1)) + 1
                shift = Tw - overlap
                for i in range(seg_num):
                    count += 1
                    start_index = i * shift
                    end_index = start_index + Tw
                    window = data[start_index:end_index, :]
                    filename = dir_name + f'{prefix}_{str(count).zfill(2)}.mat'
                    savemat(filename, {f'{prefix}_{str(count).zfill(2)}': window})
                print(f'{prefix} Split Success!')

