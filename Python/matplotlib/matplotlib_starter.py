import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.linspace(0, 2*np.pi, 400)
x_long = np.linspace(0, 20, 500)
t = np.linspace(0, 2*np.pi, 500)
X, Y = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
Z = np.sin(X**2 + Y**2)

fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
fig.suptitle("ECEN4293 Matplotlib Oscillations & Patterns",
             fontsize=14, fontweight='bold')

# (A) Sin & Cos
ax = axs[0, 0]
# TODO: plot sin and cos with legend, grid, labels

# (B) Damped Oscillation
ax = axs[0, 1]
# TODO: plot exp(-0.1x)*sin(5x)

# (C) 2D Wave Heatmap
ax = axs[1, 0]
# TODO: show Z with imshow and colorbar

# (D) Lissajous Curve
ax = axs[1, 1]
# TODO: plot sin(3t), sin(4t), set equal aspect

plt.show()
