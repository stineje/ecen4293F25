import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')  # only red dots, no connecting line

# --- Static reference curve (full sqrt(x) path) ---
x_full = np.linspace(0, 10, 100)
y_full = np.sqrt(x_full)
ax.plot(x_full, y_full, 'b--', lw=1.5, alpha=0.5, label='Reference √x')
ax.legend()


def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("y = √x")
    ax.set_title("Animated Points along $y = \sqrt{x}$")
    return ln,


def update(frame):
    xdata.append(frame)
    ydata.append(frame**0.5)
    ln.set_data(xdata, ydata)
    return ln,


ani = FuncAnimation(fig, update, frames=range(
    11), init_func=init, blit=True, repeat=False)

# --- Save as PowerPoint-friendly GIF ---
writer = PillowWriter(fps=2)   # 2 frames per second (nice and slow)
ani.save("sqrt_points_animation.gif", writer=writer)
print("Saved animation as 'sqrt_points_animation.gif'!")

plt.show()
