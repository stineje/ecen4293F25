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
# -------------------------------
def f(x):
    return 0.5 * (3*x[0]**2 + 4*x[0]*x[1] + 6*x[1]**2) - (2*x[0] - 8*x[1])

def grad_f(x):
    return np.array([
        3*x[0] + 2*x[1] - 2,    # ∂f/∂x₁
        2*x[0] + 6*x[1] + 8     # ∂f/∂x₂
    ])

# -------------------------------
# Steepest Descent (fixed step)
# -------------------------------
def steepest_descent(x0, alpha=0.05, max_iter=100, tol=1e-6):
    x = x0.copy().astype(float)
    path = [x.copy()]
    for _ in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - alpha * g
        path.append(x.copy())
    return np.array(path)

# -------------------------------
# BFGS (SciPy) with callback path
# -------------------------------
def bfgs_with_path(x0, max_iter=100, tol=1e-6):
    path = []
    def cb(xk):
        path.append(xk.copy())
    res = minimize(
        f, x0, method='BFGS', jac=grad_f, callback=cb,
        options={'gtol': tol, 'maxiter': max_iter, 'disp': False}
    )
    # Ensure start and final are included
    if len(path) == 0 or not np.allclose(path[0], x0):
        path.insert(0, x0.copy())
    if not np.allclose(path[-1], res.x):
        path.append(res.x.copy())
    return np.array(path), res

# -------------------------------
# Metrics helpers
# -------------------------------
def path_length(P):
    return np.sum(np.linalg.norm(P[1:] - P[:-1], axis=1))

def summarize(name, P):
    gnorm = np.linalg.norm(grad_f(P[-1]))
    return {
        "method": name,
        "iters": len(P) - 1,
        "f*": f(P[-1]),
        "||grad f||*": gnorm,
        "path_len": path_length(P)
    }

def print_table(rows):
    hdr = f"{'Method':<18} {'Iters':>5} {'f*':>14} {'||grad f||*':>14} {'Path length':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:<18} {r['iters']:>5d} {r['f*']:>14.6f} {r['||grad f||*']:>14.6f} {r['path_len']:>14.6f}")

# -------------------------------
# Main comparison
# -------------------------------
if __name__ == "__main__":
    x0 = np.array([3.0, -2.0])

    # Run both methods
    sd_path = steepest_descent(x0, alpha=0.05, max_iter=100, tol=1e-6)
    bfgs_path, bfgs_res = bfgs_with_path(x0, max_iter=100, tol=1e-6)

    # Print metrics
    rows = [summarize("Steepest Descent", sd_path),
            summarize("BFGS", bfgs_path)]
    print("\n=== Comparison: Steepest Descent vs. BFGS ===")
    print_table(rows)

    # Contour backdrop
    x1 = np.linspace(-1, 4, 200)
    x2 = np.linspace(-4, 4, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 0.5*(3*X1**2 + 4*X1*X2 + 6*X2**2) - (2*X1 - 8*X2)

    # --- Plot both trajectories ---
    plt.figure(figsize=(7,6))
    cs = plt.contour(X1, X2, Z, levels=30, cmap='gray')

    # Steepest Descent path (Orange)
    plt.plot(sd_path[:,0], sd_path[:,1], color='#F97306', marker='o',
             linestyle='-', linewidth=1.5, markersize=5,
             label='Steepest Descent')

    # BFGS path (Blue)
    plt.plot(bfgs_path[:,0], bfgs_path[:,1], 'bo-', label='BFGS (Blue)',
             linewidth=1.5, markersize=4)

    # Labels & aesthetics
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title('Comparison of Trajectories: Steepest Descent vs. BFGS')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    outname = "comparison_trajectory.png"
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.show()