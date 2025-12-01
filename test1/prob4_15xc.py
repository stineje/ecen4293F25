import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Helper: finite differences
# -------------------------------
def finite_differences(f, x0, h):
    """
    Compute forward, backward, and centered
    finite-difference approximations to f'(x0)
    with step size h.
    """
    forward = (f(x0 + h) - f(x0)) / h
    backward = (f(x0) - f(x0 - h)) / h
    centered = (f(x0 + h) - f(x0 - h)) / (2.0 * h)
    return forward, backward, centered

def percent_rel_error(approx, true):
    return abs((true - approx) / true) * 100.0 if true != 0 else np.nan

# -------------------------------
# Function from Problem 4.12
# -------------------------------
def f_cubic(x):
    return 25*x**3 - 6*x**2 + 7*x - 88

def f_cubic_prime(x):
    return 75*x**2 - 12*x + 7

# Additional example functions (for extra credit)
def f_sin(x):
    return np.sin(x)

def f_sin_prime(x):
    return np.cos(x)

def f_exp(x):
    return np.exp(x)

def f_exp_prime(x):
    return np.exp(x)

# -------------------------------
# Automation over functions/points/h
# -------------------------------
test_functions = [
    {"name": "cubic_4p12", "f": f_cubic, "fp": f_cubic_prime},
    {"name": "sin",        "f": f_sin,   "fp": f_sin_prime},
    {"name": "exp",        "f": f_exp,   "fp": f_exp_prime},
]

# Points and step sizes to test
x_points = [1.0, 2.0, 3.0]
h_values = [0.25, 0.1]

for func_info in test_functions:
    fname = func_info["name"]
    f = func_info["f"]
    fp = func_info["fp"]

    print(f"\n=== Function: {fname} ===")

    for x0 in x_points:
        true_val = fp(x0)
        print(f"\nAt x0 = {x0:.3f}, true derivative = {true_val:.8f}")

        for h in h_values:
            fwd, bwd, cen = finite_differences(f, x0, h)
            methods = ["Forward", "Backward", "Centered"]
            approxs = [fwd, bwd, cen]
            errors = [percent_rel_error(a, true_val) for a in approxs]

            print(f"\n  h = {h:.3f}")
            for name, approx, err in zip(methods, approxs, errors):
                print(f"    {name:8s}:  {approx:14.8f}   % rel. error = {err:10.6f}%")

            # ---- Plot errors and SAVE figure for this (f, x0, h) ----
            xpos = np.arange(len(methods))
            plt.figure(figsize=(7, 4))
            plt.bar(xpos, errors)
            plt.xticks(xpos, methods)
            plt.ylabel("True percent relative error")
            plt.title(
                f"Errors for {fname}, x0 = {x0:.2f}, h = {h:.3f}"
            )
            plt.grid(True, axis="y")
            plt.tight_layout()

            # create a safe filename
            safe_name = f"{fname}_x{str(x0).replace('.','p')}_h{str(h).replace('.','p')}"
            filename = f"prob4_15xc_{safe_name}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            
