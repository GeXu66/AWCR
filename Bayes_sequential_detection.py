import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns

# 1. 生成基础数据
np.random.seed(42)

# 已知工况的特征向量 (10个工况，每个20维)
known_patterns = np.random.randn(10, 20) * 2


# 生成测试特征向量（模拟10个不同情况）
def generate_test_patterns():
    test_patterns = []

    # 1. 纯已知工况
    test_patterns.append(known_patterns[2] + np.random.randn(20) * 0.1)  # 接近工况3
    test_patterns.append(known_patterns[5] + np.random.randn(20) * 0.1)  # 接近工况6

    # 2. 纯未知工况
    unknown_pattern1 = np.random.randn(20) * 2
    unknown_pattern2 = np.random.randn(20) * 2
    test_patterns.extend([unknown_pattern1, unknown_pattern2])

    # 3. 组合工况
    # 3.1 两个已知工况的组合
    combined_known = 0.6 * known_patterns[1] + 0.4 * known_patterns[3] + np.random.randn(20) * 0.1
    test_patterns.append(combined_known)

    # 3.2 已知和未知工况的组合
    combined_unknown = 0.7 * known_patterns[4] + 0.3 * np.random.randn(20) * 2
    test_patterns.append(combined_unknown)

    # 补充更多测试样本
    test_patterns.append(known_patterns[7] + np.random.randn(20) * 0.1)
    test_patterns.append(unknown_pattern1 * 1.2)
    test_patterns.append(0.5 * known_patterns[8] + 0.5 * known_patterns[9])
    test_patterns.append(0.8 * known_patterns[0] + 0.2 * np.random.randn(20) * 2)

    return np.array(test_patterns)


# 2. 贝叶斯序贯检验框架实现
class BayesianSequentialDetector:
    def __init__(self, known_patterns, alpha=0.05, beta=0.05):
        self.known_patterns = known_patterns
        self.n_patterns = len(known_patterns)
        self.dim = known_patterns.shape[1]
        self.alpha = alpha
        self.beta = beta

        # 初始化协方差矩阵
        self.covariance = np.eye(self.dim) * 0.1

        # 初始化先验概率
        self.priors = np.ones(self.n_patterns + 1) / (self.n_patterns + 1)  # +1 for unknown

        # 设置阈值
        self.A = (1 - beta) / alpha
        self.B = beta / (1 - alpha)

    def compute_likelihood(self, x, pattern):
        return multivariate_normal.pdf(x, pattern, self.covariance)

    def compute_unknown_likelihood(self, x):
        # 使用更宽的协方差矩阵来建模未知工况
        unknown_cov = self.covariance * 5
        return multivariate_normal.pdf(x, np.zeros_like(x), unknown_cov)

    def detect(self, x):
        likelihoods = np.array([self.compute_likelihood(x, pattern)
                                for pattern in self.known_patterns])
        unknown_likelihood = self.compute_unknown_likelihood(x)

        # 计算后验概率
        all_likelihoods = np.append(likelihoods, unknown_likelihood)
        posteriors = all_likelihoods * self.priors
        posteriors = posteriors / np.sum(posteriors)

        # 计算贝叶斯因子
        bf = likelihoods / unknown_likelihood

        # 决策规则
        max_bf = np.max(bf)
        max_bf_idx = np.argmax(bf)

        if max_bf >= self.A:
            return "Known-" + str(max_bf_idx), posteriors
        elif max_bf <= self.B:
            return "Unknown", posteriors
        else:
            return "Combined", posteriors


# 3. 运行检测
def run_detection():
    test_patterns = generate_test_patterns()
    detector = BayesianSequentialDetector(known_patterns)

    results = []
    all_posteriors = []

    for i, pattern in enumerate(test_patterns):
        result, posteriors = detector.detect(pattern)
        results.append(result)
        all_posteriors.append(posteriors)

    return results, np.array(all_posteriors)


# 4. 可视化结果
def visualize_results(results, posteriors):
    # 绘制后验概率热图
    plt.figure(figsize=(15, 8))

    # 后验概率热图
    plt.subplot(1, 2, 1)
    sns.heatmap(posteriors, cmap='YlOrRd')
    plt.title('Posterior Probabilities')
    plt.xlabel('Pattern Index (last is unknown)')
    plt.ylabel('Test Pattern Index')

    # 检测结果
    plt.subplot(1, 2, 2)
    categories = np.array([result.split('-')[0] for result in results])
    unique_categories = np.unique(categories)
    category_counts = [np.sum(categories == cat) for cat in unique_categories]

    plt.pie(category_counts, labels=unique_categories, autopct='%1.1f%%')
    plt.title('Detection Results Distribution')

    plt.tight_layout()
    plt.show()

    # 打印详细结果
    print("\nDetailed Results:")
    for i, result in enumerate(results):
        print(f"Test Pattern {i}: {result}")


# 运行完整检测流程
results, posteriors = run_detection()
visualize_results(results, posteriors)