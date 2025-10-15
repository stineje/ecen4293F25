import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0   # Cavity size (m)
U = 1.0   # Lid velocity (m/s)
k = 0.1   # Nonlinear coefficient
N = 50    # Grid points for plotting

# Grid
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)


def residual(A):
    """ Nonlinear equation residual: f(A) = A * (L^2/4)(1 + kA^2) - U/2 """
    u_mid = A * (L**2 / 4)  # u(L/2, L/2) = A * (L^2/4)
    return u_mid * (1 + k * A**2) - U / 2


def residual_derivative(A):
    return (L**2 / 4) * (1 + 3 * k * A**2)


# TODO: Students implement Newton-Raphson to find A
A = 0.5   # Initial guess
tol = 1e-6
max_iter = 20
for _ in range(max_iter):
    f = residual(A)
    f_prime = residual_derivative(A)
    if abs(f_prime) < 1e-10:  # Prevent division by zero
        print("Derivative too small, stopping Newton-Raphson")
        break
    A_new = A - f / f_prime
    if abs(A_new - A) < tol:
        print(
            f"Newton-Raphson converged to A = {A_new:.3f} after {_ + 1} iterations")
        break
    A = A_new
else:
    print("Newton-Raphson did not converge")

# Compute stream function and velocities
psi = A * X * (L - X) * Y * (L - Y)
u = A * X * (L - X) * (L - 2 * Y)
v = -A * Y * (L - Y) * (L - 2 * X)

# Ensure no NaN values
u = np.nan_to_num(u, nan=0.0)
v = np.nan_to_num(v, nan=0.0)

# Plot contour and quiver
plt.figure(figsize=(6, 6))
plt.contour(X, Y, psi, levels=10, colors='k')
plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], color='b')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title(f'2D Stokes Flow: Lid-Driven Cavity (A = {A:.3f})')
plt.axis('square')
plt.savefig('stokes_cavity_plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Plot nonlinear equation and derivative
A_range = np.linspace(0, 2, 100)
f_values = residual(A_range)
f_prime_values = residual_derivative(A_range)
plt.figure(figsize=(8, 5))
plt.plot(A_range, f_values, 'b-', label='f(A) = A (L²/4)(1 + kA²) - U/2')
plt.plot(A_range, f_prime_values, 'r--', label="f'(A)")
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(A, color='g', linestyle='--', label=f'Solution A = {A:.3f}')
plt.xlabel('A')
plt.ylabel('f(A) and f\'(A)')
plt.title('Nonlinear Equation for Stokes Flow')
plt.legend()
plt.grid(True)
plt.savefig('stokes_nonlinear_plot.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
