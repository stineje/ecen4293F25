import matplotlib.pyplot as plt
import numpy as np

# Generate random data
data = np.random.randn(1000)  # 1000 samples from normal distribution

plt.hist(data, bins=30, color="skyblue", edgecolor="black")
plt.title("Histogram Example")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig('histogram.png')
plt.show()
