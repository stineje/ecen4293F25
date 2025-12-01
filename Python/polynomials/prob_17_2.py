import numpy as np
import matplotlib.pyplot as plt

# Given data points from the table
# x: 0, 1, 2.5, 3, 4.5, 5, 6
# y: 2, 5.4375, 7.3516, 7.5625, 8.4453, 9.1875, 12
data_points = np.array([
    (0.0, 2.0),
    (1.0, 5.4375),
    (2.5, 7.3516),
    (3.0, 7.5625),
    (4.5, 8.4453),
    (5.0, 9.1875),
    (6.0, 12.0)
])

x_target = 3.5  # we want y(3.5)

# --- Step 1: order points to be centered around x_target ---
centered_data_points = data_points[np.argsort(
    np.abs(data_points[:, 0] - x_target)
)]

print("x-values ordered by closeness to 3.5:")
print(centered_data_points[:, 0])

def divided_differences(x, y):
    """Compute the divided-difference table and return Newton coefficients."""
    n = len(y)
    coef = np.zeros((n, n))
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])
    return coef[0]  # first row: coefficients for Newton form

def newton_polynomial(coef, x_data, x):
    """Evaluate Newton’s interpolating polynomial with coefficients coef at x."""
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p

# --- Step 2: build Newton polynomial using centered ordering ---
x_centered = centered_data_points[:, 0]
y_centered = centered_data_points[:, 1]

coef_centered = divided_differences(x_centered, y_centered)
y_approx = newton_polynomial(coef_centered, x_centered, x_target)

print(f"\nNewton interpolant estimate at x = {x_target}: {y_approx:.6f}")

# (Optional) Compare with original ordering to show same mathematical value
x_original = data_points[:, 0]
y_original = data_points[:, 1]
coef_original = divided_differences(x_original, y_original)
y_approx_original = newton_polynomial(coef_original, x_original, x_target)
print(f"Using original order (for comparison): {y_approx_original:.6f}")

# --- Step 3: plot interpolant and data points ---
x_plot = np.linspace(0.0, 6.0, 400)
y_plot = [newton_polynomial(coef_centered, x_centered, xv) for xv in x_plot]

plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_plot, label="Newton interpolant", linestyle="--")
plt.scatter(data_points[:, 0], data_points[:, 1],
            color="red", label="Data points")
plt.scatter(x_target, y_approx, color="blue",
            label=f"y(3.5) ≈ {y_approx:.4f}", marker="x")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Newton Interpolation centered around x = 3.5")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("prob_17_2.png", dpi=300)
plt.show()
