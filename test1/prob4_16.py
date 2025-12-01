import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Function from Problem 4.12
# ---------------------------------
def f(x):
    return 25*x**3 - 6*x**2 + 7*x - 88

# True second derivative (analytic)
def f2_true(x):
    # f'(x)  = 75x^2 - 12x + 7
    # f''(x) = 150x - 12
    return 150*x - 12

# Point where we want the second derivative
x0 = 2.0

# Step sizes
h_values = [0.2, 0.1]

true_val = f2_true(x0)

def abs_error(approx, true):
    return abs(true - approx)

def rel_error(approx, true):
    return abs((true - approx) / true)

methods = []
approxs = []
abs_errors = []
rel_errors = []

print(f"True second derivative f''(2) = {true_val:.6f}\n")

for h in h_values:
    # Centered difference approximation for f''(x0)
    # f''(x0) ≈ (f(x0 + h) - 2 f(x0) + f(x0 - h)) / h^2
    second_approx = (f(x0 + h) - 2.0*f(x0) + f(x0 - h)) / (h**2)

    ae = abs_error(second_approx, true_val)
    re = rel_error(second_approx, true_val)

    methods.append(f"h = {h}")
    approxs.append(second_approx)
    abs_errors.append(ae)
    rel_errors.append(re)

    print(f"h = {h:.3f} :  f''(2) ≈ {second_approx:12.6f}   "
          f"abs error = {ae:12.6e}   rel. error = {re:10.6e}")

# ---------------------------------
# Plot absolute and relative errors and SAVE to file
# ---------------------------------
xpos = np.arange(len(h_values))

plt.figure(figsize=(10,4))

# Absolute error
plt.subplot(1, 2, 1)
plt.bar(xpos, abs_errors)
plt.xticks(xpos, methods)
plt.ylabel("Absolute error")
plt.title("Absolute Error in f''(2) Approximation")
plt.grid(True, axis="y")

# Relative error
plt.subplot(1, 2, 2)
plt.bar(xpos, rel_errors)
plt.xticks(xpos, methods)
plt.ylabel("Percent relative error (%)")
plt.title("Percent Relative Error in f''(2) Approximation")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("prob4_16.png", dpi=300, bbox_inches="tight")
plt.show()
