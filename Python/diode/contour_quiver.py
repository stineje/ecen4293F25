import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0  # Domain size
N = 50   # Grid points

# Grid
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)

# Sample data: Gaussian function z = e^(-(x^2 + y^2)), like signal intensity
Z = np.exp(-(X**2 + Y**2))
U = -2 * X * Z  # Gradient in x-direction
V = -2 * Y * Z  # Gradient in y-direction

# Contour plot
plt.figure(figsize=(6, 6))
plt.contour(X, Y, Z, levels=10, colors='k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour Plot: Signal Intensity')
plt.axis('square')
plt.savefig('contour_plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Quiver plot
plt.figure(figsize=(6, 6))
plt.quiver(X[::2, ::2], Y[::2, ::2], U[::2, ::2], V[::2, ::2], color='b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quiver Plot: Signal Gradients')
plt.axis('square')
plt.savefig('quiver_plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Streamplot
plt.figure(figsize=(6, 6))
plt.streamplot(X, Y, U, V, color='b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Streamplot: Signal Flow')
plt.axis('square')
plt.savefig('streamplot_plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Heatmap
plt.figure(figsize=(6, 6))
plt.imshow(Z, origin='lower', extent=[-L, L, -L, L], cmap='hot')
plt.colorbar(label='Intensity')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Heatmap: Signal Intensity')
plt.axis('square')
plt.savefig('heatmap_plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
