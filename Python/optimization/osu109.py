import matplotlib.pyplot as plt
import numpy as np


def golden_ratio_spiral(n_terms):
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # Initialize lists for x and y coordinates
    x = [0]
    y = [0]

    # Initial angle
    angle = 0

    # Generate points for the spiral
    for i in range(1, n_terms):
        # Radius for the current term
        radius = phi ** i

        # Update angle
        angle += np.pi / 2

        # Calculate new coordinates
        x.append(x[-1] + radius * np.cos(angle))
        y.append(y[-1] + radius * np.sin(angle))

    return x, y


# Number of terms
n_terms = 15
x, y = golden_ratio_spiral(n_terms)

plt.figure(figsize=(8, 8))
plt.plot(x, y, marker='o', linestyle='-')
plt.title('Golden Ratio Spiral Approximation')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(True)
plt.show()
