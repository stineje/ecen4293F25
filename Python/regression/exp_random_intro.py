import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
x = np.linspace(0, 10, 20)
y = np.exp(0.3*x) + rng.normal(0, 2, size=len(x))

plt.scatter(x, y, color='orange')
plt.title("Random-looking Data: Hidden Structure?")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()

plt.savefig("random_exponential_data.png", dpi=300)
plt.show()
