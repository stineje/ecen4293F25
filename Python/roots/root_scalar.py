from scipy.optimize import root_scalar
import numpy as np

# Define your function
def f(x):
    return x**3 - x - 2

# Example 1: bracketed interval (like fzero([a, b]))
sol = root_scalar(f, bracket=[1, 2])
print(f"Root = {sol.root},  Converged? {sol.converged}")

# Example 2: single initial guess with secant method (like fzero(x0))
sol2 = root_scalar(f, x0=1.0, x1=2.0, method='secant')
print(f"Root = {sol2.root}")
