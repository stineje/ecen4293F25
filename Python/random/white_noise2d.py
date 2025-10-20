import numpy as np
import matplotlib.pyplot as plt

# --- Set up reproducible RNG ---
rng = np.random.default_rng(0)

# --- Generate 2D random data (e.g., 100x100 matrix) ---
data = rng.normal(0, 1, (100, 100))   # mean=0, std=1

plt.figure(figsize=(5,4))
plt.imshow(data, cmap='gray', origin='lower', interpolation='nearest')
plt.colorbar(label='Amplitude')
plt.title("2D White Noise Field (Normal Distribution)")
plt.tight_layout()

plt.savefig("random_field_2d.png", dpi=300)
plt.show()   
