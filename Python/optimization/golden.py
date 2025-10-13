import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def fib(n):
    """First n Fibonacci numbers starting 1,1,2,3,..."""
    a, b = 1, 1
    out = [a, b]
    for _ in range(n - 2):
        a, b = b, a + b
        out.append(b)
    return out[:n]

def draw_fibonacci_spiral(n_terms=10, linew=2.0):
    assert n_terms >= 2, "Use at least 2 terms."

    F = fib(n_terms)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#e6e6e6")  # soft gray like your example

    # --- seed with two 1×1 squares side-by-side (landscape start) ---
    squares = []  # list of (x, y, side)
    squares.append((0.0, 0.0, 1.0))  # first 1×1 at (0,0)
    squares.append((1.0, 0.0, 1.0))  # second 1×1 to the right

    # current bounding box of the tiling
    xmin, ymin = 0.0, 0.0
    xmax, ymax = 2.0, 1.0

    # directions cycle: up, left, down, right (counterclockwise growth)
    # we'll start at "up" for F[2], matching the classic layout
    DIRS = ["up", "left", "down", "right"]
    d = 0  # index into DIRS

    # --- place subsequent Fibonacci squares around the current rectangle ---
    for k in range(2, n_terms):
        s = float(F[k])

        if DIRS[d] == "up":
            # place square directly above the current rectangle
            x_new, y_new = xmin, ymax
            squares.append((x_new, y_new, s))
            ymax += s

        elif DIRS[d] == "left":
            # place square to the left
            x_new, y_new = xmin - s, ymin
            squares.append((x_new, y_new, s))
            xmin -= s

        elif DIRS[d] == "down":
            # place square below
            x_new, y_new = xmin, ymin - s
            squares.append((x_new, y_new, s))
            ymin -= s

        elif DIRS[d] == "right":
            # place square to the right
            x_new, y_new = xmax, ymin
            squares.append((x_new, y_new, s))
            xmax += s

        d = (d + 1) % 4  # rotate direction

    # --- draw the squares (thick black lines, no fill) ---
    for (x, y, s) in squares:
        ax.add_patch(Rectangle((x, y), s, s, fill=False, linewidth=linew, edgecolor="black"))

    # --- draw the quarter-circle spiral arcs inside each square ---
    #   Arc centers/angles depend on where that square was attached.
    #   We replay the placement directions to compute centers & angles.
    d = 0
    # We start arcs from the 2nd square (index 1) to keep continuity.
    # The first two 1×1 squares begin the pattern.
    # For each square placed at direction DIRS[d], the arc:
    #   center and angle range are chosen to make a smooth CCW spiral.
    # We need the previous bbox state at placement time; instead we
    # infer from the square’s own corner according to direction.

    # Reconstruct the same direction sequence for each square after the first two
    for k in range(2, len(squares)):
        x, y, s = squares[k]

        if DIRS[d] == "up":
            # center at the lower-left corner of that square
            cx, cy = x, y
            theta = np.linspace(0.0, np.pi/2, 120)

        elif DIRS[d] == "left":
            # center at the lower-right corner
            cx, cy = x + s, y
            theta = np.linspace(np.pi/2, np.pi, 120)

        elif DIRS[d] == "down":
            # center at the upper-right corner
            cx, cy = x + s, y + s
            theta = np.linspace(np.pi, 3*np.pi/2, 120)

        elif DIRS[d] == "right":
            # center at the upper-left corner
            cx, cy = x, y + s
            theta = np.linspace(3*np.pi/2, 2*np.pi, 120)

        X = cx + s * np.cos(theta)
        Y = cy + s * np.sin(theta)
        ax.plot(X, Y, color="black", linewidth=linew)

        d = (d + 1) % 4

    # --- frame the whole golden rectangle nicely ---
    ax.set_xlim(xmin - 0.02 * (xmax - xmin), xmax + 0.02 * (xmax - xmin))
    ax.set_ylim(ymin - 0.02 * (ymax - ymin), ymax + 0.02 * (ymax - ymin))
    plt.tight_layout()
    plt.savefig("fibonacci_spiral.png", dpi=300, bbox_inches='tight')
    plt.show()

# Example: bigger layout like your screenshot (go out to F_11 = 89)
draw_fibonacci_spiral(n_terms=11, linew=2.5)
