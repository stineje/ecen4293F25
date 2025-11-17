import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Rosenbrock function ---


def rosenbrock(x, y, a=1, b=100):
    """
    Computes Rosenbrock's function:
        f(x, y) = (a - x)^2 + b*(y - x^2)^2
    Default parameters: a = 1, b = 100
    """
    return (a - x)**2 + b * (y - x**2)**2


# --- Create a grid of points ---
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# --- 3D Surface Plot ---
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0, cmap='viridis')
ax1.set_title("Rosenbrock Function (3D Surface)", fontsize=13)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')
ax1.view_init(elev=30, azim=-60)
plt.tight_layout()
plt.savefig("rosenbrock_surface.png", dpi=200)
plt.show()

# --- 2D Contour Plot ---
fig2, ax2 = plt.subplots(figsize=(6, 5))
contours = ax2.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='plasma')
ax2.clabel(contours, inline=True, fontsize=8)
ax2.set_title("Rosenbrock Function (Contour Plot)", fontsize=13)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.plot(1, 1, 'r*', markersize=12, label='Global Minimum (1,1)')
ax2.legend()
plt.tight_layout()
plt.savefig("rosenbrock_contour.png", dpi=200)
plt.show()
