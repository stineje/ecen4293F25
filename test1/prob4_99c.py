"""
Convergence study for finite-difference approximations of f'(1)
for f(x) = sin(x), with step sizes h_k = 10^{-k}, k = 1,...,12.

Part (a):
    - Compute forward, backward, and centered one-point approximations
    - Plot absolute error vs h on a log-log scale
    - Report empirical slopes for coarse h

Part (b):
    - Build Richardson-extrapolated derivative D_RE from centered differences
    - Add its error curve to the same log-log plot
    - Report its empirical slope
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# -------------------------------------------------
# Setup
# -------------------------------------------------
def f(x):
    return np.sin(x)

def fprime_true_scalar(x):
    return math.cos(x)

x0 = 1.0
true_deriv = fprime_true_scalar(x0)

# h_k = 10^{-k}, k = 1..12
k_vals = np.arange(1, 13)
h_vals = 10.0 ** (-k_vals)   # array of step sizes

# Storage for approximations and errors
D_fwd = np.zeros_like(h_vals, dtype=float)
D_bwd = np.zeros_like(h_vals, dtype=float)
D_ctr = np.zeros_like(h_vals, dtype=float)

err_fwd = np.zeros_like(h_vals, dtype=float)
err_bwd = np.zeros_like(h_vals, dtype=float)
err_ctr = np.zeros_like(h_vals, dtype=float)

# -------------------------------------------------
# (a) One-point approximations: forward, backward, centered
# -------------------------------------------------
for i, h in enumerate(h_vals):
    # Forward difference
    D_fwd[i] = (math.sin(x0 + h) - math.sin(x0)) / h

    # Backward difference
    D_bwd[i] = (math.sin(x0) - math.sin(x0 - h)) / h

    # Centered difference
    D_ctr[i] = (math.sin(x0 + h) - math.sin(x0 - h)) / (2.0 * h)

    # Absolute errors
    err_fwd[i] = abs(D_fwd[i] - true_deriv)
    err_bwd[i] = abs(D_bwd[i] - true_deriv)
    err_ctr[i] = abs(D_ctr[i] - true_deriv)

# -------------------------------------------------
# (b) Richardson extrapolation based on centered differences
# D_RE = (4 D_{h/2}^{ctr} - D_h^{ctr}) / 3
# We can compute D_{h}^{ctr} and D_{h/2}^{ctr} independently.
# We'll define an h grid for Richardson with k = 1..11
# so that both h and h/2 are "reasonable" values.
# -------------------------------------------------
k_vals_RE = np.arange(1, 12)            # k = 1..11
h_vals_RE = 10.0 ** (-k_vals_RE)

D_RE = np.zeros_like(h_vals_RE, dtype=float)
err_RE = np.zeros_like(h_vals_RE, dtype=float)

for i, h in enumerate(h_vals_RE):
    Dh  = (math.sin(x0 + h)      - math.sin(x0 - h))      / (2.0 * h)
    Dh2 = (math.sin(x0 + 0.5*h)  - math.sin(x0 - 0.5*h))  / (2.0 * (0.5*h))
    D_RE[i] = (4.0 * Dh2 - Dh) / 3.0
    err_RE[i] = abs(D_RE[i] - true_deriv)

# -------------------------------------------------
# Empirical slopes on log-log scale (coarse h region)
# Choose, say, the first 5 h values (k=1..5, i.e., h=1e-1..1e-5)
# Feel free to adjust this slice if you want.
# -------------------------------------------------
def empirical_slope(h_arr, err_arr, n_points=5, label=""):
    # use first n_points where error > 0 to avoid log(0)
    h_slice = h_arr[:n_points]
    e_slice = err_arr[:n_points]

    # Filter out any zero errors (unlikely here)
    mask = e_slice > 0
    h_slice = h_slice[mask]
    e_slice = e_slice[mask]

    logh = np.log10(h_slice)
    loge = np.log10(e_slice)

    # Fit log(error) ~ m log(h) + c
    m, c = np.polyfit(logh, loge, 1)
    print(f"Empirical slope for {label:10s} ≈ {m: .4f}")
    return m

print("\n=== Empirical slopes on coarse h (first ~5 points) ===")
s_fwd  = empirical_slope(h_vals,   err_fwd, n_points=5, label="forward")
s_bwd  = empirical_slope(h_vals,   err_bwd, n_points=5, label="backward")
s_ctr  = empirical_slope(h_vals,   err_ctr, n_points=5, label="centered")
s_RE   = empirical_slope(h_vals_RE, err_RE, n_points=5, label="Richardson")

# -------------------------------------------------
# Plot |error| vs h on log-log scale
# -------------------------------------------------
plt.figure(figsize=(8, 6))

plt.loglog(h_vals,   err_fwd,  'r-o',  label="Forward diff")
plt.loglog(h_vals,   err_bwd,  'b-s',  label="Backward diff")
plt.loglog(h_vals,   err_ctr,  'g-^',  label="Centered diff")
plt.loglog(h_vals_RE, err_RE,  'k-d',  label="Richardson (centered)")

plt.xlabel("h")
plt.ylabel(r"$|D_h - f'(1)|$")
plt.title(r"Error vs step size for $f(x)=\sin x$ at $x_0=1$")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("prob4_99_sin_x.png", dpi=300)
plt.show()
