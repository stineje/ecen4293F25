"""
Finite difference approximations for

    f(x) = x^3 - 2x + 4

on the interval [-2, 2] with h = 0.25.

Now includes:
- Tables printed for all first-derivative approximations
- Tables printed for all second-derivative approximations
- Plots + savefig
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# -------------------------------------------------
# Define function and exact derivatives
# -------------------------------------------------
def f(x):
    return x**3 - 2.0*x + 4.0

def fprime_exact(x):
    return 3.0*x**2 - 2.0

def fsecond_exact(x):
    return 6.0*x

# -------------------------------------------------
# Grid setup
# -------------------------------------------------
a = -2.0
b =  2.0
h = 0.25
N = int((b - a)/h) + 1

x = a + h * np.arange(N)
fx = f(x)

# -------------------------------------------------
# 1st derivative finite differences
# -------------------------------------------------
idx1 = np.arange(1, N-1)
x1 = x[idx1]

# Approximations
fwd1  = (fx[idx1+1] - fx[idx1])       / h
bwd1  = (fx[idx1]   - fx[idx1-1])     / h
cent1 = (fx[idx1+1] - fx[idx1-1])     / (2*h)

# Exact
fprime1 = fprime_exact(x1)

# -------------------------------------------------
# Print table for first derivative
# -------------------------------------------------
print("\n================ FIRST DERIVATIVE APPROXIMATIONS ================")
print("   x        exact f'(x)     forward      backward     centered")
for i in range(len(idx1)):
    print(f"{x1[i]:7.3f}   {fprime1[i]:12.6f}   {fwd1[i]:10.6f}   {bwd1[i]:10.6f}   {cent1[i]:10.6f}")

# -------------------------------------------------
# 2nd derivative finite differences
# -------------------------------------------------
idx2 = np.arange(2, N-2)
x2 = x[idx2]

# Approximations
fwd2  = (fx[idx2+2] - 2*fx[idx2+1] + fx[idx2])         / h**2
bwd2  = (fx[idx2]   - 2*fx[idx2-1] + fx[idx2-2])       / h**2
cent2 = (fx[idx2+1] - 2*fx[idx2]   + fx[idx2-1])       / h**2

# Exact
fsecond2 = fsecond_exact(x2)

# -------------------------------------------------
# Print table for second derivative
# -------------------------------------------------
print("\n================ SECOND DERIVATIVE APPROXIMATIONS ================")
print("   x        exact f''(x)    forward      backward     centered")
for i in range(len(idx2)):
    print(f"{x2[i]:7.3f}   {fsecond2[i]:12.6f}   {fwd2[i]:10.6f}   {bwd2[i]:10.6f}   {cent2[i]:10.6f}")


# -------------------------------------------------
# Plot: First derivative approximations
# -------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(x1, fprime1, 'k-', label="Exact f'(x)")
plt.plot(x1, fwd1,   'r--', label="Forward diff")
plt.plot(x1, bwd1,   'b-.', label="Backward diff")
plt.plot(x1, cent1,  'g:',  label="Centered diff")

plt.grid(True)
plt.legend()
plt.xlabel("x")
plt.ylabel("First derivative")
plt.title("First derivative: finite differences vs exact")
plt.savefig("prob4_19_first_derivative.png", dpi=300)
plt.show()

# -------------------------------------------------
# Plot: Second derivative approximations
# -------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(x2, fsecond2, 'k-', label="Exact f''(x)")
plt.plot(x2, fwd2,     'r--', label="Forward diff (2nd)")
plt.plot(x2, bwd2,     'b-.', label="Backward diff (2nd)")
plt.plot(x2, cent2,    'g:',  label="Centered diff (2nd)")

plt.grid(True)
plt.legend()
plt.xlabel("x")
plt.ylabel("Second derivative")
plt.title("Second derivative: finite differences vs exact")
plt.savefig("prob4_19_second_derivative.png", dpi=300)
plt.show()
