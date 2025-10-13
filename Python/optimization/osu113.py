from parabolic_min import parabolic_min
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x**2/10 - 2*np.sin(x)

x1, x2, x3 = 0.0, 1.0, 4.0 

xmin, fmin, ea, n = parabolic_min(f, x1, x2, x3, Ea=1e-5, verbose=True)

print('Solution = {0:8.15g}'.format(xmin))
print('Function value at solution = {0:8.15g}'.format(fmin))
print('Relative error = {0:8.3e}'.format(ea))
print(f'Number of iterations = {n}')

x = np.linspace(min(x1, x3), max(x1, x3), 400)
plt.figure(figsize=(8,5))
plt.plot(x, f(x), label=r"$f(x)=x^2/10 - 2\sin x$")
plt.axvline(xmin, ls='--', color='r', label=f"Parabolic min @ {xmin:.6f}")
plt.plot(xmin, fmin, 'ro')
plt.legend()
plt.grid(True)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Parabolic Interpolation Minimum (Chapra)")
plt.tight_layout()
plt.savefig("parabolic_minimum_plot.png", dpi=300, bbox_inches="tight")
plt.show()
