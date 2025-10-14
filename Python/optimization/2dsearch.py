import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create grid and function
x1 = np.linspace(-2.0, 0.0, 20)
x2 = np.linspace(0.3, 20.0, 20)
X, Y = np.meshgrid(x1, x2)
Z = 2 + X - Y + 2*X**2 + 2*X*Y + Y**2

# 3D Wireframe plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, color='k')
ax.set_xticks([-2, -1.5, -1.0, -0.5, 0])
ax.set_yticks([0, 1, 2, 3])
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Wireframe Plot')
plt.savefig('wireframe_plot.png', dpi=300)
plt.close(fig)

# Contour plot
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.contour(X, Y, Z)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.grid(True)
ax2.set_title('Contour Plot')
plt.savefig('contour_plot.png', dpi=300)
plt.close(fig2)
