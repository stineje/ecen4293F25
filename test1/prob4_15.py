import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Function from Problem 4.12
# ---------------------------------
def f(x):
    return 25*x**3 - 6*x**2 + 7*x - 88

# True derivative (analytic)
def fprime_true(x):
    # f'(x) = 75x^2 - 12x + 7
    return 75*x**2 - 12*x + 7

# Point and step size
x0 = 2.0
h = 0.25

# True derivative at x0
true_val = fprime_true(x0)

# ---------------------------------
# Finite difference approximations
# ---------------------------------

# Forward difference O(h)
forward = (f(x0 + h) - f(x0)) / h

# Backward difference O(h)
backward = (f(x0) - f(x0 - h)) / h

# Centered difference O(h^2)
centered = (f(x0 + h) - f(x0 - h)) / (2*h)

# ---------------------------------
# Error calculations
# ---------------------------------
def abs_error(approx, true):
    return abs(true - approx)

def percent_rel_error(approx, true):
    return abs((true - approx) / true) * 100.0

methods = ["Forward", "Backward", "Centered"]
approxs = [forward, backward, centered]
abs_errors = [abs_error(a, true_val) for a in approxs]
rel_errors = [percent_rel_error(a, true_val) for a in approxs]

# ---------------------------------
# Print results
# ---------------------------------
print(f"True derivative f'(2) = {true_val:.6f}\n")

for name, approx, ae, re in zip(methods, approxs, abs_errors, rel_errors):
    print(f"{name:9s} : {approx:12.6f}   abs error = {ae:12.6f}   % rel. error = {re:10.6f}%")

# ---------------------------------
# Plot both absolute and relative errors
# ---------------------------------
plt.figure(figsize=(10,4))

xpos = np.arange(len(methods))

# Absolute error subplot
plt.subplot(1, 2, 1)
plt.bar(xpos, abs_errors)
plt.xticks(xpos, methods)
plt.ylabel("Absolute error")
plt.title("Absolute Error of f'(2) Approximations")
plt.grid(True, axis="y")

# Relative error subplot
plt.subplot(1, 2, 2)
plt.bar(xpos, rel_errors)
plt.xticks(xpos, methods)
plt.ylabel("Percent relative error (%)")
plt.title("Percent Relative Error of f'(2) Approximations")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("prob4_15.png", dpi=300, bbox_inches="tight")
plt.show()
