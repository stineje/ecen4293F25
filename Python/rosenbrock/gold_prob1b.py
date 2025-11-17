from parabolic_min import parabolic_min
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 4*x - 1.8*x**2 + 1.2*x**3 - 0.3*x**4


# Initial points for parabolic interpolation
xl, xu, x3 = 1.75, 2.00, 2.50

# Minimize -f to find the maximum of f; force exactly 5 iterations
argmax, g_min, ea, it = parabolic_min(
    lambda x: -f(x),
    xl, xu, x3,
    Ea=-1.0,        # negative so we DON'T early-stop on tolerance
    maxit=5,        # do exactly 5 iterations
    verbose=True
)

x_max = argmax
f_max = f(x_max)

print("\nResult after 5 parabolic interpolation iterations:")
print(f"  x_max   = {x_max:.8f}")
print(f"  f(x_max)= {f_max:.8f}")
print(f"  ea      = {ea:.3e}")
print(f"  iters   = {it:d}")

# ---- plot ----
xs = np.linspace(1.5, 2.7, 400)
ys = f(xs)
plt.figure(figsize=(7, 5))
plt.plot(xs, ys, label="f(x)")
plt.plot([xl, xu, x3], [f(xl), f(xu), f(x3)], 'kx', label="initial points")
plt.plot(x_max, f_max, 'ro', label="parabolic argmax (after 5 iters)")
plt.title("Parabolic Interpolation (maximize f by minimizing -f)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
