import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

# --- Create reproducible RNG ---
rng = np.random.default_rng(0)

# --- Generate 3D random data ---
shape = (50, 50, 50)   # (z, y, x) dimensions
data3d = rng.normal(0, 1, shape)  # white noise: mean=0, std=1

# --- Apply 3D smoothing (creates correlation) ---
corr3d = uniform_filter(data3d, size=5)

# --- Visualize central slices ---
z_mid, y_mid, x_mid = np.array(shape) // 2

fig, axs = plt.subplots(2, 3, figsize=(10, 6))
axs = axs.ravel()

# --- White noise slices ---
axs[0].imshow(data3d[z_mid,:,:], cmap='gray', origin='lower'); axs[0].set_title("White Noise (Z-slice)")
axs[1].imshow(data3d[:,y_mid,:], cmap='gray', origin='lower'); axs[1].set_title("White Noise (Y-slice)")
axs[2].imshow(data3d[:,:,x_mid], cmap='gray', origin='lower'); axs[2].set_title("White Noise (X-slice)")

# --- Correlated (smoothed) slices ---
axs[3].imshow(corr3d[z_mid,:,:], cmap='gray', origin='lower'); axs[3].set_title("Correlated (Z-slice)")
axs[4].imshow(corr3d[:,y_mid,:], cmap='gray', origin='lower'); axs[4].set_title("Correlated (Y-slice)")
axs[5].imshow(corr3d[:,:,x_mid], cmap='gray', origin='lower'); axs[5].set_title("Correlated (X-slice)")

for ax in axs:
    ax.axis('off')

plt.suptitle("3D Random Field: White vs Correlated Noise", fontsize=13)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("random_field_3d_slices.png", dpi=300)
plt.show()
