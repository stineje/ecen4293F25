# log_x2_plot.py
import numpy as np
import matplotlib.pyplot as plt

# Define x values (avoid zero because log(0) is undefined)
x = np.linspace(-5, 5, 400)
x = x[x != 0]  # remove zero

# Compute y = log(x^2)
y = np.log(x**2)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$y = \log(x^2)$", color='blue')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Plot of $y = \log(x^2)$")
plt.legend()
plt.grid(True)
plt.savefig("log_x2_plot.png", dpi=300, bbox_inches='tight')
plt.show()
