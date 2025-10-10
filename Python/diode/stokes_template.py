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
    """ TODO: Implement Newton-Raphson to find A """


# Compute stream function and velocities
psi = A * X * (L - X) * Y * (L - Y)
u = A * X * (L - X) * (L - 2 * Y)
v = -A * Y * (L - Y) * (L - 2 * X)

# Ensure no NaN values
u = np.nan_to_num(u, nan=0.0)
v = np.nan_to_num(v, nan=0.0)

# Plot contour and quiver
