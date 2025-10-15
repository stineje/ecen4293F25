import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir in sys.path:
    sys.path.remove(this_dir)   
from scipy.optimize import minimize_scalar   
sys.path.append(this_dir)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds, LinearConstraint

# -------------------------------
# Objective and gradient
# f(x0, x1) = 0.4 x0^2 - 5 x0 + x1^2 - 6 x1
# -------------------------------
def f(x):
    x0, x1 = x
    return 0.4 * x0**2 - 5.0 * x0 + x1**2 - 6.0 * x1

def grad_f(x):
    x0, x1 = x
    # ∂f/∂x0 = 0.8 x0 - 5 ; ∂f/∂x1 = 2 x1 - 6
    return np.array([0.8 * x0 - 5.0, 2.0 * x1 - 6.0])

# Unconstrained minimum (for reference): solve grad = 0
x_star_uncon = np.array([6.25, 3.0])

# -------------------------------
# Constraints
# Box bounds: 0 ≤ x0 ≤ 10 , 0 ≤ x1 ≤ 10
# Linear inequality: x1 - 0.5 x0 - 4 ≥ 0  (feasible region is above the line)
# -------------------------------
bounds = Bounds(lb=[0.0, 0.0], ub=[10.0, 10.0])
A = np.array([[-0.5, 1.0]])      # row defines: (-0.5)*x0 + 1*x1
lb = np.array([4.0])             # A @ x ≥ 4    →  x1 - 0.5*x0 ≥ 4
ub = np.array([np.inf])          # no upper limit
lin_con = LinearConstraint(A, lb, ub)

# -------------------------------
# Solve with SLSQP (or trust-constr also works)
# -------------------------------
x0_init = np.array([0.0, 0.0])
res = minimize(
    f, x0_init,
    method='SLSQP', jac=grad_f,
    bounds=bounds, constraints=[lin_con],
    options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
)

x_opt = res.x
f_opt = res.fun

# -------------------------------
# Grid & cost for plotting
# -------------------------------
x0_grid = np.arange(0.0, 10.0 + 1e-9, 0.25)
x1_grid = np.arange(0.0, 10.0 + 1e-9, 0.25)
X0, X1 = np.meshgrid(x0_grid, x1_grid)
Z = 0.4 * X0**2 - 5.0 * X0 + X1**2 - 6.0 * X1

# Constraint line: x1 = 0.5*x0 + 4
x_line = np.linspace(0.0, 10.0, 400)
y_line = 0.5 * x_line + 4.0

# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

# Contours
cs = ax.contour(X0, X1, Z, levels=20, colors='k', linewidths=0.6)
ax.clabel(cs, fmt="%.0f", fontsize=8)

# Shade the infeasible region (below the line, but within plot window)
ax.fill_between(
    x_line, 0.0, y_line, where=(y_line > 0.0),
    color='red', alpha=0.25, label='Infeasible region: $x_1 - 0.5x_0 - 4 < 0$'
)

# Draw the constraint line
ax.plot(x_line, y_line, color='tab:blue', linewidth=2.0,
        label=r'Constraint: $x_1 = 0.5\,x_0 + 4$')

# Plot start, unconstrained, and constrained optima
ax.plot(x0_init[0], x0_init[1], marker='o', color='tab:gray', markersize=7, label='Start (initial guess)')
ax.plot(x_star_uncon[0], x_star_uncon[1], marker='x', color='tab:green', markersize=9, label='Unconstrained optimum')

# Constrained optimum (orange)
ax.plot(x_opt[0], x_opt[1], marker='o', color='#F97306', markersize=9, label='Constrained optimum')

# Annotations
ax.annotate(
    rf'$x^*_c=({x_opt[0]:.3f},\,{x_opt[1]:.3f})$' + "\n" + rf'$f(x^*_c)={f_opt:.3f}$',
    xy=(x_opt[0], x_opt[1]),
    xytext=(x_opt[0]+0.5, x_opt[1]+0.5),
    arrowprops=dict(arrowstyle='->', lw=1.0), fontsize=12
)

# Axes, legend, save
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel(r'$x_0$', fontsize=14)
ax.set_ylabel(r'$x_1$', fontsize=14)
ax.set_title('Constrained Quadratic: Contours, Constraint, and Optimum', fontsize=15)
ax.legend(loc='best', frameon=True)

plt.savefig('constrained_contour.png', dpi=300, bbox_inches='tight')
print("Saved figure: constrained_contour.png")
plt.show()

# -------------------------------
# Console summary
# -------------------------------
print("\n=== Optimization Summary ===")
print(f" Success:       {res.success}  ({res.message})")
print(f" x*_constrained: [{x_opt[0]:.6f}, {x_opt[1]:.6f}]")
print(f" f(x*):         {f_opt:.6f}")
print(f" Iterations:    {res.nit}")
# Check constraint satisfaction numerically
viol = (A @ x_opt)[0] - lb[0]
print(f" Constraint value (x1 - 0.5*x0 - 4): {viol + 0:.6e}  (>= 0 means feasible)")