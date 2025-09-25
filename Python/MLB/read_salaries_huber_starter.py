#!/usr/bin/env python3
# Huber robust regression using SciPy's least_squares
# Students: complete the TODO sections

import csv
import numpy as np
from scipy import stats
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# --- Read CSV ---
def read_csv_data(file_path):
    """
    TODO:
    - Open the CSV file with `with open(file_path, mode='r', newline='')`
    - Create a csv.reader object
    - Skip the header row
    - For each row:
        * Extract Year (int), Team (str), League (str), Player (str), Salary (float)
        * Append as a dictionary to a list
    - Return the list of dictionaries
    """
    rows = []
    # TODO: open file, loop over rows, append dictionaries
    return rows

data = read_csv_data('Salaries.csv')
if not data:
    raise SystemExit("No data loaded from Salaries.csv")

# --- Load arrays from rows ---
# TODO: Read values into arrays using np.array
x = None   # TODO: np.array([...], dtype=np.float64)
y = None   # TODO: np.array([...], dtype=np.float64)

# --- Center x to improve conditioning ---
# TODO: compute xm = mean of x; xc = x - xm
xm = None   # TODO
xc = None   # TODO

# --- OLS initialization on centered x ---
# TODO: build Xc = [1, xc] and solve least squares for ac0 (centered intercept) and b0 (slope)
Xc = None             # TODO: np.column_stack([np.ones_like(xc), xc])
beta_ols = None       # TODO: np.linalg.lstsq(Xc, y, rcond=None)[0]
ac0, b0 = None, None  # TODO: unpack beta_ols

# --- Residual function for least_squares (given) ---
def resid_centered(p):
    """
    Residuals for robust regression:
      r = y - (a_c + b * x_c)
    where a_c is the intercept in centered coordinates.
    """
    ac, b = p
    return y - (ac + b * xc)

# --- Robust scale estimate from OLS residuals (given) ---
r0 = resid_centered((ac0, b0))
sigma_mad = stats.median_abs_deviation(r0, scale='normal')
if sigma_mad == 0:
    sigma_mad = np.std(r0) or 1.0

# --- Huber threshold (delta = c * sigma_mad) ---
# TODO: choose c (≈1.345 is standard) and compute delta
c = None        # TODO
delta = None    # TODO

# --- Run robust regression with SciPy ---
# TODO: call scipy.optimize.least_squares with:
#   - resid_centered as residual function
#   - x0 = np.array([ac0, b0], dtype=np.float64)
#   - loss='huber', f_scale=delta
#   - method='trf', x_scale='jac'
#   - and reasonable tolerances (ftol, xtol, gtol)
res = None      # TODO

# --- Extract solution (centered), then un-center intercept ---
ac_h, b_h = None, None        # TODO: res.x
a_h = None                    # TODO: ac_h - b_h * xm

# --- Final reporting ---
r_final = None                # TODO: y - (ac_h + b_h * xc)
sigma_final = None            # TODO: stats.median_abs_deviation(r_final, scale='normal')

print("=== SciPy Huber Regression (least_squares, centered x) ===")
print(f"Slope (robust):     {b_h}")
print(f"Intercept (robust): {a_h}")
print(f"Scale (final MAD):  {sigma_final}")
print(f"Function evals:     {res.nfev if res is not None else 'TODO'}")

# --- OLS for reference (optional, in original coordinates) ---
a_ols = None   # TODO: (ac0 - b0 * xm)
b_ols = None   # TODO: b0

# --- Plot results ---
# (These will work after the TODOs above are filled.)
x_line = np.linspace(np.min(x) if x is not None else 0,
                     np.max(x) if x is not None else 1, 200)
y_line_huber = None   # TODO:
y_line_ols = None     # TODO:

plt.figure(figsize=(8,5))
plt.scatter(x, y, s=16, alpha=0.6, label='Data')
plt.plot(x_line, y_line_huber, lw=2, label='Huber (SciPy, centered x)')
plt.plot(x_line, y_line_ols, lw=2, ls='--', label='OLS (reference)')
plt.xlabel('Year')
plt.ylabel('Salary [$]')
plt.title('Salary vs Year — Huber Robust Regression (SciPy least_squares)')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.savefig('salaries_huber_fit_scipy_centered.png', dpi=150)
plt.show()
