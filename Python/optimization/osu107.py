import matplotlib.pyplot as plt


def plot_golden_rectangles(n):
    """
    Plot a series of golden rectangles.

    Parameters:
    n (int): The number of golden rectangles to plot.
    """
    # Define the golden ratio
    golden_ratio = (1 + 5 ** 0.5) / 2

    # Initialize the figure
    fig, ax = plt.subplots()

    # Starting width and height for the largest rectangle
    width, height = 1, golden_ratio

    # Starting position for the rectangle (bottom-left corner)
    x, y = 0, 0

    # Plot rectangles
    for i in range(n):
        # Draw the rectangle
        ax.add_patch(plt.Rectangle((x, y), width, height,
                     fill=None, edgecolor='blue', linewidth=2))

        # Determine the next rectangle's size and position
        if i % 2 == 0:  # Alternating between decreasing width and height
            width, height = height - width, width
            x += width
        else:
            width, height = width, height - width
            y += height

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Set plot limits
    plt.xlim(-0.1, 2 * golden_ratio)
    plt.ylim(-0.1, 2 * golden_ratio)

    # Show the plot
    plt.title(f'Golden Rectangles (n={n})')
    plt.show()


# Example: Plot 5 golden rectangles
plot_golden_rectangles(5)
