import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir in sys.path:
    sys.path.remove(this_dir)   
from scipy.optimize import minimize_scalar   
sys.path.append(this_dir)

import numpy as np                 
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

g  = 9.81   # m/s^2
v0 = 55.0   # m/s
m  = 80.0   # kg
c  = 15.0   # kg/s
z0 = 100.0  # m

# Altitude function z(t) from the problem for bungee from Chapra
def z(t):
    return z0 + (m/c)*(v0 + (m*g)/c)*(1 - np.exp(-t/(m/c))) - (m*g/c)*t

# We minimize the negative to find the maximum altitude
f = lambda t: -z(t)

# Search in the same window used before
res = minimize_scalar(f, bracket=(0.0, 8.0), tol=1e-7, options={'maxiter': 200})

tmin = res.x               # time at which f is minimum -> z is maximum
zmax = z(tmin)             # maximum altitude
fmin = res.fun             # value of objective (-zmax)

print(f"Time at maximum altitude = {tmin:0.5f} s")
print(f"Maximum altitude         = {zmax:0.6f} m")
print(f"Objective value (−z)     = {fmin:0.6f}")
print(f"Success indicator        = {res.success}")
print(f"Number of iterations     = {res.nit}")
print(f"Function evaluations     = {res.nfev}")
