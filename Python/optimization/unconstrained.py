import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir in sys.path:
    sys.path.remove(this_dir)   
from scipy.optimize import minimize_scalar   
sys.path.append(this_dir)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -------------------------------
# Objective and gradient
# f(x0,x1) = 0.4 x0^2 - 5 x0 + x1^2 - 6 x1
# -------------------------------
def objective(x):
    x0, x1 = x
    return 0.4 * x0**2 - 5.0 * x0 + x1**2 - 6.0 * x1

def grad_objective(x):
    x0, x1 = x
    # ∂f/∂x0 = 0.8 x0 - 5 , ∂f/∂x1 = 2 x1 - 6
    return np.array([0.8 * x0 - 5.0, 2.0 * x1 - 6.0])

# Analytic unconstrained minimum (for reference): x* = (6.25, 3.0)
x_star_uncon = np.array([6.25, 3.0])

# -------------------------------
# Solve (with simple box bounds)
# -------------------------------
bounds = [(0.0, 10.0), (0.0, 10.0)]
x0_init = np.array([0.0, 0.0])

res = minimize(
    objective, x0_init,
    method='L-BFGS-B', jac=grad_objective,
    bounds=bounds,
    options={'maxiter': 200, 'disp': False}
)

x_opt = res.x
f_opt = res.fun

# -------------------------------
# Grid for plots
# -------------------------------
x0 = np.arange(0.0, 10.0 + 1e-9, 0.25)
x1 = np.arange(0.0, 10.0 + 1e-9, 0.25)
X0, X1 = np.meshgrid(x0, x1)
Z = 0.4 * X0**2 - 5.0 * X0 + X1**2 - 6.0 * X1

# -------------------------------
# (1) Contour plot
# -------------------------------
fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

# Filled contours + lines for readability
cf = ax.contourf(X0, X1, Z, levels=30, alpha=0.85, cmap='Greys')
cs = ax.contour(X0, X1, Z, levels=15, colors='k', linewidths=0.6)
ax.clabel(cs, fmt="%.0f", fontsize=8)

# Start point and optimal point (orange)
ax.plot(x0_init[0], x0_init[1], marker='o', markersize=7,
        label='Start', color='tab:blue')
ax.plot(x_opt[0], x_opt[1], marker='o', markersize=9,
        label='Optimal (L-BFGS-B)', color='#F97306')

# Optional: show unconstrained minimum as reference (if different from bounded)
ax.plot(x_star_uncon[0], x_star_uncon[1], marker='x', markersize=9,
        label='Unconstrained minimum', color='tab:green')

# Labels, legend, and annotation
ax.set_xlabel(r'$x_0$', fontsize=14)
ax.set_ylabel(r'$x_1$', fontsize=14)
ax.set_title('Quadratic Cost Contours and Optimal Point', fontsize=15)
ax.legend(loc='best', frameon=True)

# Annotate the found optimum
ax.annotate(
    rf'$x^* = ({x_opt[0]:.3f}, {x_opt[1]:.3f})$' + "\n" + rf'$f(x^*) = {f_opt:.3f}$',
    xy=(x_opt[0], x_opt[1]),
    xytext=(x_opt[0] + 0.5, x_opt[1] + 0.5),
    arrowprops=dict(arrowstyle="->", lw=1),
    fontsize=12
)

plt.savefig('cost_contour.png', dpi=300, bbox_inches='tight')
print("Saved contour to: cost_contour.png")
plt.show()

# -------------------------------
# (2) 3D surface plot
# -------------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig3 = plt.figure(figsize=(8, 6), constrained_layout=True)
ax3 = fig3.add_subplot(111, projection='3d')

surf = ax3.plot_surface(X0, X1, Z, cmap='viridis', edgecolor='none', alpha=0.9)
ax3.set_xlabel(r'$x_0$', fontsize=12)
ax3.set_ylabel(r'$x_1$', fontsize=12)
ax3.set_zlabel('Cost', fontsize=12)
ax3.set_title('Quadratic Cost Surface', fontsize=14)

# Optimal point in 3D (orange)
ax3.scatter(x_opt[0], x_opt[1], objective(x_opt), s=80, color='#F97306', label='Optimal')
ax3.legend(loc='best')

# Modest viewing angle for clarity
ax3.view_init(elev=25, azim=-135)

plt.savefig('cost_surface.png', dpi=300, bbox_inches='tight')
print("Saved surface to: cost_surface.png")
plt.show()

# -------------------------------
# Console summary (version-safe)
# -------------------------------
METHOD_USED = "L-BFGS-B"  # we passed this to minimize()

def _res_get(obj, key, default=None):
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)

print("\n=== Optimization Summary ===")
print(f" Method:        {_res_get(res, 'method', METHOD_USED)}")
print(f" Success:       {res.success}  ({res.message})")
print(f" x*:            [{x_opt[0]:.6f}, {x_opt[1]:.6f}]")
print(f" f(x*):         {f_opt:.6f}")
print(f" Iterations:    {res.nit}")

# Prefer solver-provided gradient if available; otherwise compute ours
if _res_get(res, 'jac') is not None:
    grad_norm = np.linalg.norm(res.jac)
else:
    grad_norm = np.linalg.norm(grad_objective(x_opt))
print(f" Grad norm:     {grad_norm:.3e}")

# Optional extras (exist only in some solvers/versions)
nfev = _res_get(res, 'nfev')
njev = _res_get(res, 'njev')
if nfev is not None: print(f" Function evals: {nfev}")
if njev is not None: print(f" Gradient evals: {njev}")
