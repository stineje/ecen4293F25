"""
===========================================================
Animated Lissajous Curve Demo using Matplotlib
===========================================================

This program demonstrates how to use Matplotlib to animate
a classic mathematical curve called a Lissajous figure.

Background:
-----------
A Lissajous curve is generated when two sine waves with
different frequencies are combined in orthogonal directions:

    x(t) = A * sin(a * t + \delta)
    y(t) = B * sin(b * t)

- A and B are amplitudes
- a and b are integer frequency ratios
- \delta is the phase difference

These curves are visually striking and appear in physics
(oscilloscopes, resonance), art, and even music visualization.
By adjusting the frequency ratio (a:b), the curve forms
different looping patterns.

This demo:
----------
- Uses Matplotlib’s animation API (`FuncAnimation`)
- Draws the curve point by point for a smooth animated effect
- Highlights how simple trigonometric equations can produce
  complex and beautiful visuals
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters for the Lissajous curve
A, B = 1, 1      # amplitudes
a, b = 3, 2      # frequencies
delta = np.pi/2  # phase shift

t = np.linspace(0, 2*np.pi, 1000)

x = A * np.sin(a * t + delta)
y = B * np.sin(b * t)

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.set_title("Animated Lissajous Curve")


def init():
    line.set_data([], [])
    return line,


def update(frame):
    # Draw up to the current frame
    line.set_data(x[:frame], y[:frame])
    return line,


ani = animation.FuncAnimation(
    fig, update, frames=len(t),
    init_func=init, blit=True, interval=10
)

plt.show()
