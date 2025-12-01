"""
Richardson Extrapolation & Finite Differences for g(x) = e^x sin(x)

Part 1: On [-2, 2] with h=0.25:
    - Compute forward, backward, and centered approximations for g'(x) and g''(x)
    - Plot them against the exact derivatives
    - Print max errors for each approximation

Part 2: At x0 = 2.067 with h = 0.4 and 0.2:
    - Do Richardson extrapolation for g'(x0) and g''(x0)
    - Report absolute error and true percent relative error
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# -------------------------------------------------
# g(x) and exact derivatives
# -------------------------------------------------
def g(x):
    return np.exp(x) * np.sin(x)

def gprime_true(x):
    # g'(x) = e^x (sin x + cos x)
    return np.exp(x) * (np.sin(x) + np.cos(x))

def gsecond_true(x):
    # g''(x) = e^x (2 cos x)
    return np.exp(x) * (2.0 * np.cos(x))


# -------------------------------------------------
# Generic finite-difference helpers (scalar x0)
# -------------------------------------------------
def D_centered_scalar(func, x0, h):
    """Centered difference approximation to f'(x0)."""
    return (func(x0 + h) - func(x0 - h)) / (2.0 * h)

def D2_threepoint_scalar(func, x0, h):
    """Three-point formula approximation to f''(x0)."""
    return (func(x0 + h) - 2.0 * func(x0) + func(x0 - h)) / (h * h)

def richardson(Dh, Dh2, p):
    """
    Richardson extrapolation combining approximations with steps h and h/2:
        D_rich = (2^p * D(h/2) - D(h)) / (2^p - 1)
    """
    return (2**p * Dh2 - Dh) / (2**p - 1)


def abs_err(approx, truth):
    return abs(approx - truth)

def pct_true_rel_err(approx, truth):
    return 0.0 if truth == 0 else 100.0 * abs(approx - truth) / abs(truth)


# -------------------------------------------------
# PART 1: Finite differences for g over [-2, 2], h=0.25
# -------------------------------------------------
a = -2.0
b =  2.0
h = 0.25
N = int((b - a)/h) + 1

x = a + h * np.arange(N)    # grid
gx = g(x)                   # g(x) on the grid

# ---------- First derivative: forward, backward, centered ----------
# We evaluate all three on the common interior set: indices 1..N-2
idx1 = np.arange(1, N-1)
x1 = x[idx1]

# forward at x_i: (g_{i+1} - g_i)/h
g1_fwd  = (gx[idx1 + 1] - gx[idx1]) / h

# backward at x_i: (g_i - g_{i-1})/h
g1_bwd  = (gx[idx1] - gx[idx1 - 1]) / h

# centered at x_i: (g_{i+1} - g_{i-1})/(2h)
g1_cent = (gx[idx1 + 1] - gx[idx1 - 1]) / (2.0 * h)

# exact first derivative
g1_exact = gprime_true(x1)

# errors
err1_fwd  = np.abs(g1_fwd  - g1_exact)
err1_bwd  = np.abs(g1_bwd  - g1_exact)
err1_cent = np.abs(g1_cent - g1_exact)

print("\n=== First derivative g'(x) on [-2,2], h=0.25 ===")
print(f"max |forward - exact|  = {np.max(err1_fwd):.6e}")
print(f"max |backward - exact| = {np.max(err1_bwd):.6e}")
print(f"max |centered - exact| = {np.max(err1_cent):.6e}")

# ---------- Second derivative: forward, backward, centered ----------
# For second derivative, need i-2, i-1, i, i+1, i+2 for all three formulas.
idx2 = np.arange(2, N-2)
x2 = x[idx2]

# forward second derivative: (g_{i+2} - 2g_{i+1} + g_i)/h^2
g2_fwd = (gx[idx2 + 2] - 2.0*gx[idx2 + 1] + gx[idx2]) / h**2

# backward second derivative: (g_i - 2g_{i-1} + g_{i-2})/h^2
g2_bwd = (gx[idx2] - 2.0*gx[idx2 - 1] + gx[idx2 - 2]) / h**2

# centered second derivative: (g_{i+1} - 2g_i + g_{i-1})/h^2
g2_cent = (gx[idx2 + 1] - 2.0*gx[idx2] + gx[idx2 - 1]) / h**2

# exact second derivative
g2_exact = gsecond_true(x2)

# errors
err2_fwd  = np.abs(g2_fwd  - g2_exact)
err2_bwd  = np.abs(g2_bwd  - g2_exact)
err2_cent = np.abs(g2_cent - g2_exact)

print("\n=== Second derivative g''(x) on [-2,2], h=0.25 ===")
print(f"max |forward - exact|  = {np.max(err2_fwd):.6e}")
print(f"max |backward - exact| = {np.max(err2_bwd):.6e}")
print(f"max |centered - exact| = {np.max(err2_cent):.6e}")

# ---------- Plots for g'(x) ----------
plt.figure(figsize=(10, 6))
plt.plot(x1, g1_exact, 'k-',  label="Exact g'(x)")
plt.plot(x1, g1_fwd,   'r--', label="Forward diff")
plt.plot(x1, g1_bwd,   'b-.', label="Backward diff")
plt.plot(x1, g1_cent,  'g:',  label="Centered diff")

plt.xlabel("x")
plt.ylabel("First derivative")
plt.title("First derivative of g(x) = e^x sin x\nFinite differences vs. exact, h=0.25")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("prob4_99b_first_derivative.png", dpi=300)

# ---------- Plots for g''(x) ----------
plt.figure(figsize=(10, 6))
plt.plot(x2, g2_exact, 'k-',  label="Exact g''(x)")
plt.plot(x2, g2_fwd,   'r--', label="Forward diff (2nd)")
plt.plot(x2, g2_bwd,   'b-.', label="Backward diff (2nd)")
plt.plot(x2, g2_cent,  'g:',  label="Centered diff (2nd)")

plt.xlabel("x")
plt.ylabel("Second derivative")
plt.title("Second derivative of g(x) = e^x sin x\nFinite differences vs. exact, h=0.25")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("prob4_99b_second_derivative.png", dpi=300)

# -------------------------------------------------
# PART 2: Richardson extrapolation for g at x0 = 2.067
# -------------------------------------------------
x0 = 2.067
h_big = 0.4
h_small = 0.2

# True values at x0
gp_true = gprime_true(x0)
gpp_true = gsecond_true(x0)

# First derivative approximations
Dg_h   = D_centered_scalar(lambda x: math.exp(x) * math.sin(x), x0, h_big)
Dg_h2  = D_centered_scalar(lambda x: math.exp(x) * math.sin(x), x0, h_small)
Dg_rich = richardson(Dg_h, Dg_h2, p=2)

# Second derivative approximations
Sg_h   = D2_threepoint_scalar(lambda x: math.exp(x) * math.sin(x), x0, h_big)
Sg_h2  = D2_threepoint_scalar(lambda x: math.exp(x) * math.sin(x), x0, h_small)
Sg_rich = richardson(Sg_h, Sg_h2, p=2)

def line(name, val, truth):
    print(f"{name:16s} = {val:12.8f} | "
          f"abs err = {abs_err(val, truth):10.6e} | "
          f"% true rel err = {pct_true_rel_err(val, truth):10.6e}")

print("\n=== Richardson Extrapolation for g'(2.067) ===")
line("True g'(2.067)", gp_true, gp_true)
line("D(h=0.4)",       Dg_h,    gp_true)
line("D(h=0.2)",       Dg_h2,   gp_true)
line("D_rich",         Dg_rich, gp_true)

print("\n=== Richardson Extrapolation for g''(2.067) ===")
line("True g''(2.067)", gpp_true, gpp_true)
line("S(h=0.4)",        Sg_h,      gpp_true)
line("S(h=0.2)",        Sg_h2,     gpp_true)
line("S_rich",          Sg_rich,   gpp_true)
plt.savefig("prob4_99b_error.png", dpi=300)
plt.show()
