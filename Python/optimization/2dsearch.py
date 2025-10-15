import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Define function
def f(X, Y):
    return 2 + X - Y + 2*X**2 + 2*X*Y + Y**2

# Create grid
x1 = np.linspace(-2.0, 0.0, 41)
x2 = np.linspace(0.3, 3.0, 41)
X, Y = np.meshgrid(x1, x2)
Z = f(X, Y)

# ---- 3D Wireframe plot ----
fig1 = plt.figure(figsize=(5, 4))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='k', linewidth=0.7)
ax1.view_init(elev=25, azim=-135)
ax1.set_xlim(-2.0, 0.0)
ax1.set_ylim(0.0, 3.0)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_title('Wireframe Plot')
fig1.tight_layout()
fig1.savefig('wireframe_plot.png', dpi=300, bbox_inches='tight')

# ---- Contour plot ----
fig2 = plt.figure(figsize=(5, 4))
ax2 = fig2.add_subplot(111)
levels = np.linspace(Z.min(), Z.min() + 6.0, 9)
ax2.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.8)
ax2.set_xlim(-2.0, 0.0)
ax2.set_ylim(0.0, 3.0)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Contour Plot')
ax2.grid(True, linewidth=0.6, alpha=0.6)
fig2.tight_layout()
fig2.savefig('contour_plot.png', dpi=300, bbox_inches='tight')
plt.show()