import numpy as np
import matplotlib.pyplot as plt

# --- Objective function ---
def f(x):
    return 0.5 * (3*x[0]**2 + 4*x[0]*x[1] + 6*x[1]**2) - (2*x[0] - 8*x[1])

# --- Gradient (partial derivatives) ---
def grad_f(x):
    return np.array([
        3*x[0] + 2*x[1] - 2,    # ∂f/∂x₁
        2*x[0] + 6*x[1] + 8     # ∂f/∂x₂
    ])

# --- Initialization ---
x = np.array([3.0, -2.0])
alpha = 0.05
max_iter = 20
tol = 1e-6
path = [x.copy()]

print("=== Steepest Descent Optimization ===")
print(f"{'Iter':>4s} | {'x1':>8s} | {'x2':>8s} | {'f(x)':>10s} | {'||grad f||':>10s}")
print("-"*54)

for k in range(max_iter):
    g = grad_f(x)
    fval = f(x)
    grad_norm = np.linalg.norm(g)
    print(f"{k:4d} | {x[0]:8.4f} | {x[1]:8.4f} | {fval:10.6f} | {grad_norm:10.6f}")
    
    if grad_norm < tol:
        print("\nConverged successfully.")
        break

    x = x - alpha * g
    path.append(x.copy())

print("\nFinal result:")
print(f"x* = [{x[0]:.6f}, {x[1]:.6f}]")
print(f"f(x*) = {f(x):.6f}")

# --- Create contour plot to visualize trajectory ---
x1 = np.linspace(-1, 4, 100)
x2 = np.linspace(-4, 4, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = 0.5*(3*X1**2 + 4*X1*X2 + 6*X2**2) - (2*X1 - 8*X2)

path = np.array(path)

plt.figure(figsize=(6,5))
plt.contour(X1, X2, Z, levels=30, cmap='gray')

# OSU-style orange line and dots
plt.plot(path[:,0], path[:,1], color='#F97306', marker='o',
         linestyle='-', linewidth=1.5, markersize=5,
         label='Steepest Descent Path')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Steepest Descent Trajectory')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig("steepest_descent_trajectory.png", dpi=300, bbox_inches='tight')
plt.show()