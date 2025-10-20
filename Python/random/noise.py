import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
x = np.linspace(0, 10, 25)
y = np.exp(0.3*x) + rng.normal(0, 2, len(x))

plt.scatter(x, y, color='orange')
plt.title("Pseudo-Random Noise: Reproducible Every Run")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.savefig("pseudorandom_example.png", dpi=300)
