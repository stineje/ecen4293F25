import numpy as np
import matplotlib.pyplot as plt

# --- Rosenbrock function ---


def rosenbrock(x, y, a=1, b=100):
    """Compute Rosenbrock's function f(x, y) = (a - x)^2 + b*(y - x^2)^2"""
    return (a - x)**2 + b * (y - x**2)**2


# --- Create grid ---
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# --- Main figure ---
plt.figure(figsize=(7, 5))

# Use imshow for smooth color background
plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='viridis', aspect='auto')

# Overlay contour lines (logarithmic spacing to show valley detail)
levels = np.logspace(-1, 3, 20)
contours = plt.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.7)
plt.clabel(contours, inline=True, fontsize=8)

# Mark global minimum
plt.plot(1, 1, 'r*', markersize=12, label='Global Minimum (1, 1)')

# Labels and title
plt.title("Rosenbrock Function f(x, y) = (1 - x)² + 100(y - x²)²")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.colorbar(label="Function Value f(x, y)")

plt.tight_layout()
plt.savefig("rosenbrock_2d.png", dpi=200)
plt.show()
