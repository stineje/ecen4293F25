import matplotlib.pyplot as plt
import numpy as np


def fibonacci_spiral(n_terms):
    # Generate Fibonacci sequence
    fib = [0, 1]
    for _ in range(2, n_terms):
        fib.append(fib[-1] + fib[-2])

    # Initialize lists for x and y coordinates
    x = [0]
    y = [0]

    # Generate points for the spiral
    angle = 0
    for i in range(1, n_terms):
        angle += np.pi / 2
        radius = fib[i]
        x.append(x[-1] + radius * np.cos(angle))
        y.append(y[-1] + radius * np.sin(angle))

    return x, y


# Number of Fibonacci terms
n_terms = 15
x, y = fibonacci_spiral(n_terms)

plt.figure(figsize=(8, 8))
plt.plot(x, y, marker='o', linestyle='-')
plt.title('Fibonacci Spiral')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(True)
plt.show()
