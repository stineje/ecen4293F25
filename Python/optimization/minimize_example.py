import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir in sys.path:
    sys.path.remove(this_dir)   
from scipy.optimize import minimize_scalar   
sys.path.append(this_dir)

from scipy.optimize import minimize

def f(x):
    x1 = x[0]
    x2 = x[1]
    return 2 + x1 - x2 + 2*x1**2 + 2*x1*x2 + x2**2

# Initial guess
x0 = [-0.5, 0.5]

# Perform minimization using the Nelder-Mead algorithm
result = minimize(f, x0, method='Nelder-Mead', options={'disp': True})

# Extract the minimizing values
xval = result.x

print("\nOptimal values:")
print(f"x1 = {xval[0]:.6f}")
print(f"x2 = {xval[1]:.6f}")
print(f"Minimum function value = {result.fun:.6f}")
print(f"Iterations = {result.nit}, Function evaluations = {result.nfev}")