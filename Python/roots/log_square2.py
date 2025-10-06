# log_x2_plot_with_y03.py
import numpy as np
import matplotlib.pyplot as plt

# Define x values (avoid zero because log(0) is undefined)
x = np.linspace(-5, 5, 400)
x = x[x != 0]

# Compute y = log(x^2)
y = np.log(x**2)

# Target y value
y_target = 0.3
x_target = np.exp(y_target / 2)  # positive solution
x_points = [-x_target, x_target]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r"$y = \log(x^2)$", color='blue')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')

# Plot the horizontal line y = 0.3
plt.axhline(y_target, color='red', linestyle='--', label=r"$y = 0.3$")

# Mark intersection points
plt.scatter(x_points, [y_target, y_target], color='red', zorder=5)
for xi in x_points:
    plt.text(xi, y_target + 0.2, f"x = {xi:.3f}", ha='center', color='red')

# Labels and title
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Plot of $y = \log(x^2)$ with $y = 0.3$")
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("log_x2_y03_plot.png", dpi=300, bbox_inches='tight')

print("x values where y = 0.3:", x_points)
print("Plot saved as log_x2_y03_plot.png")
