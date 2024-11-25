import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 1. 生成10个已知工况的特征向量 (20维)
np.random.seed(42)
known_patterns = np.random.randn(10, 20)
# 标准化每个特征向量
known_patterns = known_patterns / np.linalg.norm(known_patterns, axis=1)[:, np.newaxis]


# 2. 生成10个测试特征向量，包含三种情况
def generate_test_vectors():
    # 情况1：已知工况 (略微加入噪声)
    known_case = known_patterns[2] + 0.1 * np.random.randn(20)

    # 情况2：完全未知工况
    unknown_case = np.random.randn(20)

    # 情况3：已知工况的组合 (工况0和工况1的混合)
    combined_case = 0.6 * known_patterns[0] + 0.4 * known_patterns[1] + 0.1 * np.random.randn(20)

    # 生成更多测试向量
    test_vectors = []

    # 添加3个已知工况(带噪声)
    for i in [0, 3, 5]:
        vec = known_patterns[i] + 0.1 * np.random.randn(20)
        test_vectors.append(vec)

    # 添加3个未知工况
    for _ in range(3):
        vec = np.random.randn(20)
        vec = vec / np.linalg.norm(vec)
        test_vectors.append(vec)

    # 添加4个混合工况
    mix1 = 0.7 * known_patterns[7] + 0.3 * known_patterns[8] + 0.1 * np.random.randn(20)
    mix2 = 0.5 * known_patterns[4] + 0.5 * unknown_case + 0.1 * np.random.randn(20)
    mix3 = 0.8 * known_patterns[9] + 0.2 * np.random.randn(20)
    mix4 = 0.4 * known_patterns[2] + 0.4 * known_patterns[6] + 0.2 * np.random.randn(20)

    test_vectors.extend([mix1, mix2, mix3, mix4])

    # 标准化所有测试向量
    test_vectors = np.array(test_vectors)
    test_vectors = test_vectors / np.linalg.norm(test_vectors, axis=1)[:, np.newaxis]

    return test_vectors


# 3. M-ary Matched Filter实现
class MaryMatchedFilter:
    def __init__(self, known_patterns, η1=0.85, η2=1.3):
        self.known_patterns = known_patterns
        self.η1 = η1  # 最大相关峰值阈值
        self.η2 = η2  # 最大次大比值阈值

    def correlate(self, test_vector):
        # 计算与所有已知模式的相关性
        correlations = np.array([np.correlate(test_vector, pattern)[0]
                                 for pattern in self.known_patterns])
        return correlations

    def classify(self, test_vector):
        correlations = self.correlate(test_vector)
        max_corr = np.max(correlations)
        max_idx = np.argmax(correlations)

        # 找到次大相关值
        sorted_corr = np.sort(correlations)
        second_max = sorted_corr[-2]

        # 判决规则
        if max_corr < self.η1:
            return "Unknown Pattern"
        elif max_corr / second_max < self.η2:
            return f"Mixed Pattern (max correlation with Pattern {max_idx})"
        else:
            return f"Known Pattern {max_idx}"

    def get_correlation_details(self, test_vector):
        correlations = self.correlate(test_vector)
        return correlations


# 4. 测试和可视化
def visualize_results(detector, test_vectors):
    plt.figure(figsize=(15, 10))

    for i, test_vector in enumerate(test_vectors):
        correlations = detector.get_correlation_details(test_vector)
        result = detector.classify(test_vector)

        plt.subplot(5, 2, i + 1)
        plt.bar(range(len(correlations)), correlations)
        plt.title(f'Test Vector {i}: {result}')
        plt.xlabel('Pattern Index')
        plt.ylabel('Correlation')
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# 主程序
test_vectors = generate_test_vectors()
detector = MaryMatchedFilter(known_patterns)

# 打印每个测试向量的判别结果
print("Classification Results:")
for i, test_vector in enumerate(test_vectors):
    result = detector.classify(test_vector)
    correlations = detector.get_correlation_details(test_vector)
    max_corr = np.max(correlations)
    print(f"\nTest Vector {i}:")
    print(f"Result: {result}")
    print(f"Max correlation: {max_corr:.3f}")

# 可视化结果
visualize_results(detector, test_vectors)