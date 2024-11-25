import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以获得可重复的结果
np.random.seed(0)

# 生成时间轴
t = np.linspace(0, 1, 100)

# 生成三个随机信号
signal1 = np.random.normal(size=100)
signal2 = np.random.normal(size=100)
signal3 = np.random.normal(size=100)

# 绘制三个信号
plt.figure()
plt.plot(t, signal1, label='Signal 1')
plt.plot(t, signal2, label='Signal 2')
plt.plot(t, signal3, label='Signal 3')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Random Signals')
plt.legend()
plt.grid(True)
plt.show()

# 计算频谱图
fft1 = np.fft.fft(signal1)
fft2 = np.fft.fft(signal2)
fft3 = np.fft.fft(signal3)

# 计算频率轴
freq = np.fft.fftfreq(100, d=(t[1] - t[0]))

# 绘制频谱图
plt.figure()
plt.plot(freq[:50], np.abs(fft1[:50]), label='FFT of Signal 1')
plt.plot(freq[:50], np.abs(fft2[:50]), label='FFT of Signal 2')
plt.plot(freq[:50], np.abs(fft3[:50]), label='FFT of Signal 3')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum')
plt.legend()
plt.grid(True)
plt.show()