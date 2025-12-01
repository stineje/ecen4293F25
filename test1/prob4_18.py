"""
Taylor expansion for f(x) = x - 1 - 0.5*sin(x) about a = pi/2

Goal:
Determine the (lowest) Taylor order n such that the maximum error on [0, pi]
satisfies max|f(x) - T_n(x)| <= 0.015, and STOP once that order is found.

Also:
- Print each Taylor term for each tested order.
- Plot f(x) and T_0, T_1, ..., T_n (for the final n).
- Plot the error for the final n (and n-1 if it exists).
- Save all figures using savefig().
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# -----------------------------
# Function and parameters
# -----------------------------
def f_np(x):
    return x - 1.0 - 0.5 * np.sin(x)

a = math.pi / 2.0      # expansion point
tol = 0.015            # error tolerance
Nmax = 20              # max order to consider

# -----------------------------
# Analytic derivatives at a
# -----------------------------
def f_derivative_at_a(k):
    """
    Return f^(k)(a) for f(x) = x - 1 - 0.5*sin(x),
    evaluated at a = pi/2.
    """
    if k == 0:
        return a - 1.0 - 0.5 * math.sin(a)

    if k == 1:
        return 1.0 - 0.5 * math.cos(a)

    # For k >= 2, only -0.5*sin(x) contributes.
    # The derivatives of sin(x) cycle every 4:
    #   d^k/dx^k sin(x) =
    #       sin(x)   if k mod 4 == 0
    #       cos(x)   if k mod 4 == 1
    #      -sin(x)   if k mod 4 == 2
    #      -cos(x)   if k mod 4 == 3
    r = k % 4
    if r == 0:
        trig = math.sin(a); sign = 1.0
    elif r == 1:
        trig = math.cos(a); sign = 1.0
    elif r == 2:
        trig = math.sin(a); sign = -1.0
    else:  # r == 3
        trig = math.cos(a); sign = -1.0

    # Original function has -0.5 * sin(x), so multiply by -0.5
    return -0.5 * sign * trig

# -----------------------------
# Taylor polynomial T_n(x)
# -----------------------------
def taylor_T(x, n):
    """
    Evaluate the n-th order Taylor polynomial of f(x)
    about x = a at points x.
    """
    x_arr = np.array(x, dtype=float)
    h = x_arr - a
    T = np.zeros_like(x_arr)

    for k in range(n + 1):
        deriv = f_derivative_at_a(k)
        coeff = deriv / math.factorial(k)
        T += coeff * (h ** k)

    return T

# -----------------------------
# Search for order and STOP at first n with error <= tol
# -----------------------------
theta = np.linspace(0.0, math.pi, 3001)
ftrue = f_np(theta)

best_n = None
best_err = None

for n in range(Nmax + 1):
    print("\n========================")
    print(f" Taylor Series Order n = {n}")
    print("========================")

    # Print each term of the Taylor series for this n
    for k in range(n + 1):
        deriv = f_derivative_at_a(k)
        coeff = deriv / math.factorial(k)
        print(
            f"Term k={k}:  f^{k}(a) = {deriv:.6f},  "
            f"coefficient = {coeff:.6f} * (x - a)^{k}"
        )

    # Compute max error on [0, pi]
    Tvals = taylor_T(theta, n)
    err = np.max(np.abs(ftrue - Tvals))
    print(f"Max error on [0, π]: {err:.6f}")

    # Check tolerance and STOP at the first order that meets it
    if err <= tol:
        best_n = n
        best_err = err
        print("\n*** Tolerance reached. Stopping search. ***")
        break

# If no n met the tolerance, we won't plot anything fancy
if best_n is None:
    print(f"\nNo n ≤ {Nmax} satisfies max error ≤ {tol}.")
else:
    print("\n====================================")
    print(f"Lowest order n with max error ≤ {tol} is n = {best_n}")
    print(f"Max error at this order: {best_err:.6f}")
    print("====================================")

    # --------------------------------
    # Build Taylor polynomials T_0,...,T_best_n for plotting
    # --------------------------------
    T_list = [taylor_T(theta, k) for k in range(best_n + 1)]

    # --------------------------------
    # Plot f(x) and all T_k up to best_n
    # --------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(theta, ftrue, c='k', label='true f(x)')

    for k, Tk in enumerate(T_list):
        plt.plot(theta, Tk, label=f"T{k} (order {k})")

    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Function value")
    plt.title(f"f(x) and Taylor polynomials up to order n = {best_n}")
    plt.savefig("prob4_18.png", dpi=300)

    # --------------------------------
    # Error plot for best_n (and best_n-1 if it exists)
    # --------------------------------
    plt.figure(figsize=(10, 6))

    # Error for best_n
    err_best = np.abs(ftrue - T_list[best_n])
    plt.plot(theta, err_best, label=f"Error for T{best_n}", linestyle='-')

    # Optionally, error for previous order
    if best_n >= 1:
        err_prev = np.abs(ftrue - T_list[best_n - 1])
        plt.plot(theta, err_prev, label=f"Error for T{best_n-1}", linestyle='--')

    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.title(f"Error of Taylor approximations (up to n = {best_n})")
    plt.savefig("prob4_18_error.png", dpi=300)
    plt.show()
