import os, sys
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir in sys.path:
    sys.path.remove(this_dir)   
from scipy.optimize import minimize_scalar   
sys.path.append(this_dir)

import numpy as np                 
import matplotlib.pyplot as plt
from goldmin import goldmin
from bisect import bisect          

def f(x):
    return x**2 / 10 - 2 * np.sin(x)

xl = 0
xu = 4
res = minimize_scalar(f, method='golden', bracket=(xl, xu), tol=1.0e-5)
# Use Chapra
xmin, fmin, ea, n = goldmin(f, xl, xu, Ea=1.0e-5)

print("-" * 90)
print("Golden Section Search Comparison (SciPy vs Custom goldmin)")
print("-" * 90)
print(f"{'Method':<20}\t{'x_min':>12}\t{'f_min':>15}\t{'Error':>15}\t{'Iters/Evals':>15}")
print("-" * 90)
print(f"{'Chapra goldmin':<20}\t{xmin:12.8f}\t{fmin:15.8f}\t{ea:15.3e}\t{n:15d}")
print(f"{'SciPy minimize_scalar':<20}\t{res.x:12.8f}\t{res.fun:15.8f}\t{1.0e-5:15.3e}\t{res.nfev:15d}")
print("-" * 90)

print("\nInformation on difference:")
print("Both methods use the golden ratio \\phi \\approx 1.618 to iteratively reduce the bracket size.")
print("SciPy’s version is highly optimized but conceptually identical to the method in Chapra.")
print("Each iteration reuses one interior point, requiring only ONE new function evaluation per step.")
print("Observe the close match in x_min and f_min — numerical methods validate one another.\n")

# --- Visualization ---
x = np.linspace(xl, xu, 400)
y = f(x)

plt.figure(figsize=(8,5))
plt.plot(x, y, label=r"$f(x) = x^2/10 - 2\sin(x)$", color="navy")
plt.axvline(res.x, color="red", linestyle="--", label=f"SciPy min @ x={res.x:.4f}")
plt.axvline(xmin, color="orange", linestyle=":", label=f"Custom min @ x={xmin:.4f}")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Golden Section Search Comparison (SciPy vs Custom)")
plt.legend()
plt.grid(True)
plt.savefig("golden_section_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
