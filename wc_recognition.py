import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from scipy.stats import multivariate_normal


class OperatingModeAnalysis:
    def __init__(self, n_modes=10, n_features=20):
        self.n_modes = n_modes
        self.n_features = n_features
        # 生成已知工况的特征向量
        np.random.seed(42)
        self.known_modes = np.random.randn(n_modes, n_features)
        # 归一化
        self.known_modes = self.known_modes / np.linalg.norm(self.known_modes, axis=1).reshape(-1, 1)

    def generate_test_vectors(self, n_samples=10):
        test_vectors = []
        labels = []

        # 1. 生成已知工况的测试向量(添加噪声)
        for i in range(3):
            mode_idx = np.random.randint(0, self.n_modes)
            noise = np.random.normal(0, 0.1, self.n_features)
            vector = self.known_modes[mode_idx] + noise
            vector = vector / np.linalg.norm(vector)
            test_vectors.append(vector)
            labels.append(f"Known Mode {mode_idx}")

        # 2. 生成未知工况的测试向量
        for i in range(3):
            vector = np.random.randn(self.n_features)
            vector = vector / np.linalg.norm(vector)
            test_vectors.append(vector)
            labels.append("Unknown Mode")

        # 3. 生成混合工况的测试向量
        for i in range(4):
            # 随机选择两个已知工况或一个已知一个未知
            if np.random.random() < 0.5:
                # 两个已知工况的混合
                mode_idx1 = np.random.randint(0, self.n_modes)
                mode_idx2 = np.random.randint(0, self.n_modes)
                ratio = np.random.random()
                vector = ratio * self.known_modes[mode_idx1] + (1 - ratio) * self.known_modes[mode_idx2]
                vector = vector / np.linalg.norm(vector)
                test_vectors.append(vector)
                labels.append(f"Mixed Known {mode_idx1}&{mode_idx2}")
            else:
                # 已知和未知工况的混合
                mode_idx = np.random.randint(0, self.n_modes)
                unknown_vector = np.random.randn(self.n_features)
                unknown_vector = unknown_vector / np.linalg.norm(unknown_vector)
                ratio = np.random.random()
                vector = ratio * self.known_modes[mode_idx] + (1 - ratio) * unknown_vector
                vector = vector / np.linalg.norm(vector)
                test_vectors.append(vector)
                labels.append(f"Mixed Known{mode_idx}&Unknown")

        return np.array(test_vectors), labels

    def calculate_similarities(self, test_vectors):
        # 计算与所有已知工况的余弦相似度
        similarities = cosine_similarity(test_vectors, self.known_modes)
        return similarities

    def analyze_modes(self, test_vectors, threshold=0.9):
        similarities = self.calculate_similarities(test_vectors)
        max_similarities = np.max(similarities, axis=1)

        # 计算每个测试向量的熵（用于判断是否为混合工况）
        entropy = -np.sum(similarities * np.log2(np.clip(similarities, 1e-10, 1)), axis=1)

        # 归一化熵
        entropy_normalized = entropy / np.log2(self.n_modes)

        return similarities, max_similarities, entropy_normalized

    def visualize_results(self, test_vectors, labels):
        similarities, max_similarities, entropy = self.analyze_modes(test_vectors)

        # 创建图形
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

        # 1. 热力图显示相似度矩阵
        sns.heatmap(similarities, ax=ax1, cmap='YlOrRd')
        ax1.set_title('Similarity Matrix')
        ax1.set_xlabel('Known Modes')
        ax1.set_ylabel('Test Samples')

        # 2. 最大相似度变化曲线
        ax2.plot(max_similarities, 'bo-', label='Max Similarity')
        ax2.axhline(y=0.9, color='r', linestyle='--', label='Threshold')
        ax2.set_title('Maximum Similarity for Each Test Sample')
        ax2.set_xlabel('Test Sample Index')
        ax2.set_ylabel('Maximum Similarity')
        ax2.legend()

        # 3. 熵值变化曲线
        ax3.plot(entropy, 'go-', label='Normalized Entropy')
        ax3.set_title('Normalized Entropy for Each Test Sample')
        ax3.set_xlabel('Test Sample Index')
        ax3.set_ylabel('Normalized Entropy')
        ax3.legend()

        # 添加标签说明
        for i, label in enumerate(labels):
            ax2.annotate(label, (i, max_similarities[i]), textcoords="offset points",
                         xytext=(0, 10), ha='center', rotation=0)

        plt.tight_layout()
        plt.show()


# 运行示例
analyzer = OperatingModeAnalysis()
test_vectors, labels = analyzer.generate_test_vectors()
analyzer.visualize_results(test_vectors, labels)