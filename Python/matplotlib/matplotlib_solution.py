#!/usr/bin/env python3
"""
Matplotlib In-Class: Oscillations & Patterns — SOLUTION
Creates a 2×2 figure:
(A) sin & cos, (B) damped oscillation, (C) 2D wave heatmap, (D) Lissajous curve.
"""

import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.linspace(0, 2*np.pi, 400)
x_long = np.linspace(0, 20, 500)
t = np.linspace(0, 2*np.pi, 500)
X, Y = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
Z = np.sin(X**2 + Y**2)

fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
fig.suptitle("ECEN4293 Matplotlib Oscillations & Patterns",
             fontsize=14, fontweight='bold')

# (A) Sin & Cos
ax = axs[0, 0]
ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, np.cos(x), label="cos(x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("(A) Sin & Cos")
ax.grid(True, linestyle=":")
ax.legend()

# (B) Damped Oscillation
ax = axs[0, 1]
y = np.exp(-0.1*x_long) * np.sin(5*x_long)
ax.plot(x_long, y)
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.set_title("(B) Damped Oscillation")
ax.grid(True, linestyle=":")

# (C) 2D Wave Heatmap
ax = axs[1, 0]
im = ax.imshow(Z, extent=[-3, 3, -3, 3], origin='lower', aspect='auto')
cb = fig.colorbar(im, ax=ax)
cb.set_label("Value")
ax.set_title("(C) 2D Wave Pattern")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# (D) Lissajous Curve
ax = axs[1, 1]
x_l = np.sin(3*t)
y_l = np.sin(4*t)
ax.plot(x_l, y_l, lw=2)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("(D) Lissajous Curve")

# Optional export
fig.savefig("ecen4293_patterns_hi-res.png", dpi=300)
fig.savefig("ecen4293_patterns.pdf")

plt.show()
