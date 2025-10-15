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
# 1) x1 - 0.5 x0 - 4 ≥ 0  → A1 = [-0.5, 1],  A1 @ x ≥ 4
# 2) x1 + 0.3 x0 - 9 ≥ 0  → A2 = [ 0.3, 1],  A2 @ x ≥ 9
# -------------------------------
bounds = Bounds(lb=[0.0, 0.0], ub=[10.0, 10.0])
A = np.array([[-0.5, 1.0],
              [ 0.3, 1.0]])
lb = np.array([4.0, 9.0])
ub = np.array([np.inf, np.inf])
lin_cons = LinearConstraint(A, lb, ub)

# A feasible initial guess helps SLSQP
x0_init = np.array([6.0, 8.0])  # satisfies both constraints

# -------------------------------
# Solve
# -------------------------------
res = minimize(
    f, x0_init,
    method='SLSQP', jac=grad_f,
    bounds=bounds, constraints=[lin_cons],
    options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
)
x_opt = res.x
f_opt = res.fun

# -------------------------------
# Grid for contours and constraint lines
# -------------------------------
x0_grid = np.arange(0.0, 10.0 + 1e-9, 0.25)
x1_grid = np.arange(0.0, 10.0 + 1e-9, 0.25)
X0, X1 = np.meshgrid(x0_grid, x1_grid)
Z = 0.4 * X0**2 - 5.0 * X0 + X1**2 - 6.0 * X1

# Constraint lines in (x0,x1):
x_line = np.linspace(0.0, 10.0, 400)
y1 = 0.5 * x_line + 4.0           # x1 = 0.5 x0 + 4
y2 = -0.3 * x_line + 9.0          # x1 = -0.3 x0 + 9

# Intersection of the two lines (for reference/labeling)
# 0.5 x + 4 = -0.3 x + 9  -> 0.8 x = 5 -> x=6.25 ; y=7.125
x_int = 6.25
y_int = 0.5 * x_int + 4.0

# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

# Cost contours
cs = ax.contour(X0, X1, Z, levels=20, colors='k', linewidths=0.6)
ax.clabel(cs, fmt="%.0f", fontsize=8)

# Shade infeasible region (below either constraint line, within axes limits)
ax.fill_between(x_line, 0.0, y1, color='red', alpha=0.18,
                label=r'Infeasible: $x_1 - 0.5x_0 - 4 < 0$')
ax.fill_between(x_line, 0.0, y2, color='red', alpha=0.18,
                label=r'Infeasible: $x_1 + 0.3x_0 - 9 < 0$')

# Draw constraint lines
ax.plot(x_line, y1, color='tab:blue', lw=2.0,
        label=r'$x_1 = 0.5\,x_0 + 4$')
ax.plot(x_line, y2, color='tab:green', lw=2.0,
        label=r'$x_1 = -0.3\,x_0 + 9$')

# Markers: start, unconstrained optimum, constrained optimum
ax.plot(x0_init[0], x0_init[1], marker='o', color='tab:gray',
        markersize=7, label='Start (feasible)')
ax.plot(x_star_uncon[0], x_star_uncon[1], marker='x', color='tab:purple',
        markersize=9, label='Unconstrained optimum')

# Constrained optimum (orange)
ax.plot(x_opt[0], x_opt[1], marker='o', color='#F97306',
        markersize=9, label='Constrained optimum')

# Optional: intersection point of constraints
ax.plot(x_int, y_int, marker='d', color='black', markersize=6,
        label='Constraint intersection')

# Annotate solution
ax.annotate(
    rf'$x^*_c=({x_opt[0]:.3f},\,{x_opt[1]:.3f})$' + "\n" + rf'$f(x^*_c)={f_opt:.3f}$',
    xy=(x_opt[0], x_opt[1]),
    xytext=(x_opt[0] + 0.6, x_opt[1] + 0.6),
    arrowprops=dict(arrowstyle='->', lw=1.0), fontsize=12
)

# Axes, legend, save
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel(r'$x_0$', fontsize=14)
ax.set_ylabel(r'$x_1$', fontsize=14)
ax.set_title('Quadratic with Two Linear Inequality Constraints', fontsize=15)
ax.legend(loc='best', frameon=True)

plt.savefig('constrained_two_constraints.png', dpi=300, bbox_inches='tight')
print("Saved figure: constrained_two_constraints.png")
plt.show()

# -------------------------------
# Console summary
# -------------------------------
print("\n=== Optimization Summary ===")
print(f" Success:        {res.success}  ({res.message})")
print(f" x*_constrained: [{x_opt[0]:.6f}, {x_opt[1]:.6f}]")
print(f" f(x*):          {f_opt:.6f}")
print(f" Iterations:     {res.nit}")

# Check constraint values (should be >= 0)
c1_val = x_opt[1] - 0.5 * x_opt[0] - 4.0
c2_val = x_opt[1] + 0.3 * x_opt[0] - 9.0
print(f" c1 = x1 - 0.5*x0 - 4   = {c1_val:.6e}  (>= 0 feasible)")
print(f" c2 = x1 + 0.3*x0 - 9   = {c2_val:.6e}  (>= 0 feasible)")