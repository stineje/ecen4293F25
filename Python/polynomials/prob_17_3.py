import numpy as np
import matplotlib.pyplot as plt

# Given data points
data_points = np.array([
    (0, 0.5),
    (1, 3.134),
    (2, 5.3),
    (5.5, 9.9),
    (11, 10.2),
    (13, 9.35),
    (16, 7.2),
    (18, 6.2)
])

x_target = 8.0

# Step 1: Sort points around x = 8 to maximize interpolation accuracy
centered_data_points = data_points[np.argsort(np.abs(data_points[:, 0] - x_target))]


def divided_differences_table(x, y):
    """
    Build the full divided-difference table.
    Returns an n x n array 'coef' where
      coef[i, 0] = f(x_i)
      coef[i, 1] = f[x_i, x_{i+1}]
      ...
    The Newton coefficients are the first row coef[0, :].
    """
    n = len(y)
    coef = np.zeros((n, n))
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])
    return coef


def newton_polynomial(coef_row, x_data, x):
    """
    Evaluate Newton's interpolating polynomial at x using
    the coefficients from the first row of the divided-difference table.
    """
    n = len(coef_row) - 1
    p = coef_row[n]
    for k in range(1, n + 1):
        p = coef_row[n - k] + (x - x_data[n - k]) * p
    return p


def print_divided_diff_table(title, x, table):
    """
    Pretty-print the divided-difference table.
    Only the upper triangle (i <= n-1-j) is meaningful.
    """
    n = len(x)
    print("\n" + title)
    print("-" * (12 * (n + 1)))
    header = ["x_i", "f[x_i]"] + [f"Δ^{j}f" for j in range(1, n)]
    print("".join(f"{h:>12s}" for h in header))
    for i in range(n):
        row = [f"{x[i]:.4g}", f"{table[i,0]:.6g}"]
        for j in range(1, n):
            if i <= n - 1 - j:
                row.append(f"{table[i,j]:.6g}")
            else:
                row.append("")  # blank for entries below the "staircase"
        print("".join(f"{r:>12s}" for r in row))


# --- Centered ordering ---
x_centered = centered_data_points[:, 0]
y_centered = centered_data_points[:, 1]
table_centered = divided_differences_table(x_centered, y_centered)
coef_centered = table_centered[0, :]       # first row = Newton coefficients
estimate_centered = newton_polynomial(coef_centered, x_centered, x_target)

# --- Original ordering ---
x_original = data_points[:, 0]
y_original = data_points[:, 1]
table_original = divided_differences_table(x_original, y_original)
coef_original = table_original[0, :]
estimate_original = newton_polynomial(coef_original, x_original, x_target)

# Print both tables
print_divided_diff_table("Divided-Difference Table (Centered Ordering)", x_centered, table_centered)
print_divided_diff_table("Divided-Difference Table (Original Ordering)", x_original, table_original)

print(f"\nEstimate at x = {x_target} (centered ordering) : {estimate_centered:.6f}")
print(f"Estimate at x = {x_target} (original ordering) : {estimate_original:.6f}")

# --- Plotting the results ---
x_vals = np.linspace(0, 18, 200)
y_centered_vals = [newton_polynomial(coef_centered, x_centered, xv) for xv in x_vals]
y_original_vals = [newton_polynomial(coef_original, x_original, xv) for xv in x_vals]

plt.figure(figsize=(12, 6))
plt.plot(x_vals, y_centered_vals,
         label='Centered Data Interpolation', linestyle='--')
plt.plot(x_vals, y_original_vals,
         label='Original Order Interpolation', linestyle=':')
plt.scatter(data_points[:, 0], data_points[:, 1],
            color='red', label='Data Points')
plt.scatter(x_target, estimate_centered, color='blue',
            label=f'Estimate (Centered): {estimate_centered:.4f}', marker='x')
plt.scatter(x_target, estimate_original, color='green',
            label=f'Estimate (Original): {estimate_original:.4f}', marker='x')

plt.legend()
plt.xlabel('x')
plt.ylabel('Interpolated y')
plt.title('Newton Interpolation Comparison at x = 8')
plt.grid(True)
plt.tight_layout()
plt.savefig("prob_17_3.png", dpi=300)
print("\nSaved figure: prob_17_3.png")
plt.show()
