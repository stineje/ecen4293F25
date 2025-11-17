from goldmin_print import goldmin_print
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 4*x - 1.8*x**2 + 1.2*x**3 - 0.3*x**4


# Golden-section search parameters
xl = -2
xu = 4
Es_percent = 1.0                 # 1% relative stopping criterion
Ea = Es_percent / 100.0          # goldmin expects a fraction

# Minimize the negative to find the maximum of f
x_star_neg, fmin_neg, ea, n = goldmin_print(lambda x: -f(x), xl, xu, Ea=Ea)

# Translate back to the maximum of the original f
xmax = x_star_neg
fmax = f(xmax)

print('\n=== Result for maximizing f on [-2, 4] with Es = 1% ===')
print('x at maximum        = {0:8.15g}'.format(xmax))
print('f(x) at maximum     = {0:8.15g}'.format(fmax))
print('Relative error frac = {0:8.3e}'.format(ea))
print('Relative error  %   = {0:8.3e}'.format(ea*100))
print('Iterations          = {0:5d}'.format(n))

# Plot the function on [xl, xu] and mark the maximum
x = np.linspace(xl, xu, 400)
y = f(x)

plt.figure(figsize=(7, 5))
plt.plot(x, y, 'b-', label='f(x)')
plt.plot(xmax, fmax, 'ro', label='Maximum')
plt.title('Golden-Section Search (maximize f via minimizing -f), Es = 1%')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
