import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_golden_rectangles(n):
    # Set up the plot
    fig, ax = plt.subplots()

    # Starting width and height
    # height is the golden ratio times the width
    width, height = 1, (1 + 5 ** 0.5) / 2

    # Plot rectangles
    for i in range(n):
        # Draw the rectangle
        rect = patches.Rectangle(
            (0, 0), width, height, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)

        # Adjust dimensions for next rectangle
        width, height = height, width + height  # Golden ratio calculation

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Set plot limits
    plt.xlim(-1, height)
    plt.ylim(-1, height)

    # Show the plot
    plt.title(f'Golden Rectangles (n={n})')
    plt.show()


# Example: Plot 5 golden rectangles
plot_golden_rectangles(8)
