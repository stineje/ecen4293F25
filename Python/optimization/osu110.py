import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def fibonacci_rectangles(n_terms):
    # Generate Fibonacci sequence
    fib = [0, 1]
    for _ in range(2, n_terms):
        fib.append(fib[-1] + fib[-2])

    # Initialize lists for rectangle corners
    rectangles = []
    x_start = 0
    y_start = 0
    angle = 0

    for i in range(1, n_terms):
        width = fib[i]
        height = fib[i-1]

        if angle % 360 == 0:
            # Rectangle going up
            rectangles.append((x_start, y_start, width, height))
            y_start += height
        elif angle % 360 == 90:
            # Rectangle going right
            rectangles.append(
                (x_start - width, y_start - height, width, height))
            x_start -= width
        elif angle % 360 == 180:
            # Rectangle going down
            rectangles.append((x_start, y_start - height, width, height))
            y_start -= height
        elif angle % 360 == 270:
            # Rectangle going left
            rectangles.append((x_start, y_start, width, height))
            x_start += width

        angle += 90

    return rectangles


# Number of Fibonacci terms
n_terms = 15
rectangles = fibonacci_rectangles(n_terms)

plt.figure(figsize=(10, 10))

for rect in rectangles:
    x, y, width, height = rect
    plt.gca().add_patch(patches.Rectangle((x, y), width,
                                          height, edgecolor='black', facecolor='none'))

# Plot settings
plt.title('Golden Rectangles using Fibonacci Tiling')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(True)
plt.show()
