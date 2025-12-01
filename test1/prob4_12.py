import numpy as np
import matplotlib.pyplot as plt

# Original function
def f(x):
    return 25*x**3 - 6*x**2 + 7*x - 88

# Derivatives
def f1(x):
    return 75*x**2 - 12*x + 7

def f2(x):
    return 150*x - 12

def f3(x):
    return 150     # constant

# Base point and evaluation point
x0 = 1.0
x_eval = 3.0
h = x_eval - x0

# True value
f_true = f(x_eval)

# Taylor approximations
T0 = f(x0)
T1 = f(x0) + f1(x0)*h
T2 = f(x0) + f1(x0)*h + (f2(x0)/2.0)*(h**2)
T3 = f(x0) + f1(x0)*h + (f2(x0)/2.0)*(h**2) + (f3(x0)/6.0)*(h**3)

approxs = [T0, T1, T2, T3]

# Percent relative errors
errors = [abs((f_true - Tn)/f_true)*100 for Tn in approxs]

# Print results
print(f"True value f(3) = {f_true:.6f}\n")
for n, (Tn, err) in enumerate(zip(approxs, errors)):
    print(f"Order n = {n}:  T_{n}(3) = {Tn:.6f},  true % relative error = {err:.4f}%")

# ---- Plot and SAVE to file ----
orders = np.array([0, 1, 2, 3])
plt.figure(figsize=(7,4))
plt.bar(orders, errors)
plt.xticks(orders, [r"$T_0$", r"$T_1$", r"$T_2$", r"$T_3$"])
plt.ylabel("True percent relative error at x = 3")
plt.title("Error of Taylor Approximations of f(x) at x = 3 (about x0 = 1)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("prob4_12.png", dpi=300, bbox_inches="tight")
plt.show()

