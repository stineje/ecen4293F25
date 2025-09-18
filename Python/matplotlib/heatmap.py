import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 600)
y = np.linspace(-5, 5, 600)
X, Y = np.meshgrid(x, y)

# Interference pattern: two waves at different orientations
Z = np.sin(3*X + 4*Y) + np.sin(3*X - 4*Y)

plt.imshow(Z, extent=[-5, 5, -5, 5], origin='lower',
           cmap='plasma', aspect='auto')
plt.colorbar(label="Intensity")
plt.title("Interference Heatmap")
plt.show()
