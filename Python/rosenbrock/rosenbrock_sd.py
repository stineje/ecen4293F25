import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def rosenbrock(x):
    """Rosenbrock's function f(x, y) = (1 - x)^2 + 100*(y - x^2)^2"""
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2


def grad_rosenbrock(x):
    """Gradient of the Rosenbrock function"""
    dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])


def steepest_descent(f, grad, x0, alpha=1e-3, tol=1e-6, max_iter=10000):
    x = x0.copy()
    path = [x.copy()]
    fvals = [f(x)]
    for i in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - alpha * g
        path.append(x.copy())
        fvals.append(f(x))
    return np.array(path), np.array(fvals), i+1  # include iteration count


def run_bfgs(x0):
    result = minimize(rosenbrock, x0, method='BFGS', jac=grad_rosenbrock,
                      options={'disp': False, 'return_all': True})
    path = np.array(result['allvecs'])
    fvals = np.array([rosenbrock(p) for p in path])
    return path, fvals, result.nit, result.x, result.fun


if __name__ == "__main__":
    # Starting point
    x0 = np.array([-1.5, 2.0])

    # Run both algorithms
    path_sd, fvals_sd, iters_sd = steepest_descent(
        rosenbrock, grad_rosenbrock, x0, alpha=0.001)
    path_bfgs, fvals_bfgs, iters_bfgs, sol_bfgs, fmin_bfgs = run_bfgs(x0)

    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100*(Y - X**2)**2

    plt.figure(figsize=(8, 6))
    levels = np.logspace(-1, 3, 20)
    plt.contour(X, Y, Z, levels=levels, cmap='plasma')
    plt.plot(path_sd[:, 0], path_sd[:, 1], 'r--',
             lw=1.8, label='Steepest Descent Path')
    plt.plot(path_bfgs[:, 0], path_bfgs[:, 1], 'b-', lw=2.0, label='BFGS Path')
    plt.plot(1, 1, 'g*', markersize=12, label='Global Minimum (1, 1)')
    plt.title("Rosenbrock Function Optimization Paths")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rosenbrock_paths.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.semilogy(fvals_sd, 'r--', label='Steepest Descent')
    plt.semilogy(fvals_bfgs, 'b-', label='BFGS')
    plt.xlabel('Iteration')
    plt.ylabel('f(x, y)')
    plt.title('Convergence Comparison on Rosenbrock Function')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.6)
    plt.tight_layout()
    plt.savefig("rosenbrock_convergence.png", dpi=200)
    plt.show()

    # -----------------------------------------------------------
    # Print Results Summary
    # -----------------------------------------------------------
    print("\n==================== Optimization Summary ====================")
    print(f"Initial Point: x0 = {x0}")
    print("--------------------------------------------------------------")
    print("Steepest Descent:")
    print(f"  Iterations   : {iters_sd}")
    print(f"  Final x      : ({path_sd[-1,0]:.6f}, {path_sd[-1,1]:.6f})")
    print(f"  Final f(x,y) : {fvals_sd[-1]:.6e}")
    print("--------------------------------------------------------------")
    print("BFGS:")
    print(f"  Iterations   : {iters_bfgs}")
    print(f"  Final x      : ({sol_bfgs[0]:.6f}, {sol_bfgs[1]:.6f})")
    print(f"  Final f(x,y) : {fmin_bfgs:.6e}")
    print("==============================================================\n")
