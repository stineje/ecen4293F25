import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(0)
steps = 500
x = np.cumsum(np.random.choice([-1, 1], steps))
y = np.cumsum(np.random.choice([-1, 1], steps))

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_title("Random Walk Animation")


def init():
    line.set_data([], [])
    return line,


def update(i):
    line.set_data(x[:i], y[:i])
    return line,


ani = animation.FuncAnimation(
    fig, update, frames=steps, init_func=init, blit=True, interval=30)
plt.show()
