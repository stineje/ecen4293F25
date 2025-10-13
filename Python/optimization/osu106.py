import matplotlib.pyplot as plt
import numpy as np


def plot_fibonacci_spiral(n):
    # Generate Fibonacci sequence up to n elements
    fibonacci_sequence = [0, 1]
    for i in range(2, n):
        next_number = fibonacci_sequence[-1] + fibonacci_sequence[-2]
        fibonacci_sequence.append(next_number)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Initial coordinates and size
    x, y = 0, 0
    angle = 0

    # Plot squares and arcs for the Fibonacci spiral
    for i in range(2, n):
        # Current Fibonacci number is the length of the square's side
        side_length = fibonacci_sequence[i]

        # Draw the square
        ax.add_patch(plt.Rectangle((x, y), side_length,
                     side_length, fill=None, edgecolor='blue'))

        # Calculate the arc parameters
        theta = np.linspace(np.radians(angle), np.radians(angle + 90), 100)
        arc_x = x + fibonacci_sequence[i - 1] + side_length * np.cos(theta)
        arc_y = y + fibonacci_sequence[i - 1] + side_length * np.sin(theta)
        ax.plot(arc_x, arc_y, color='red')

        # Update position and angle for the next square
        if angle == 0:
            x += side_length
            y += 0
        elif angle == 90:
            x += 0
            y += side_length
        elif angle == 180:
            x -= fibonacci_sequence[i - 1]
            y += side_length - fibonacci_sequence[i - 2]
        elif angle == 270:
            x -= fibonacci_sequence[i - 2]
            y -= fibonacci_sequence[i - 1]

        # Update the angle for the next arc
        angle = (angle + 90) % 360

    # Set aspect ratio and plot limits
    ax.set_aspect('equal')
    plt.xlim(-fibonacci_sequence[-1], fibonacci_sequence[-1] * 2)
    plt.ylim(-fibonacci_sequence[-1], fibonacci_sequence[-1] * 2)

    # Show the plot
    plt.title(f'Fibonacci Spiral with {n} Squares')
    plt.show()


# Example: Plot Fibonacci spiral with 10 squares
plot_fibonacci_spiral(10)
