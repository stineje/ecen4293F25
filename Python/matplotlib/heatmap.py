import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)
Z = np.sin(X**2 + Y**2)

plt.imshow(Z, extent=(-3, 3, -3, 3), origin='lower', cmap='plasma')
plt.colorbar(label="Value")
plt.title("Heatmap of sin(x² + y²)")
plt.show()
