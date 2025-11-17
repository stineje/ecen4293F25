from goldmin_error import goldmin_error
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**2/10 - 2*np.sin(x)


xl = 0
xu = 4

# Run golden-section search with precomputed iteration count
xmin, fmin, ea, n = goldmin_error(f, xl, xu, Ea=1.0e-4)

# Print results
print('Solution = {0:8.15g}'.format(xmin))
print('Function value at solution = {0:8.15g}'.format(fmin))
print('Absolute error bound = {0:8.3e}'.format(ea))
print('Number of iterations = {0:5d}'.format(n))

# --- Plot the function and the minimum found ---
x = np.linspace(xl, xu, 400)
y = f(x)

plt.figure(figsize=(7, 5))
plt.plot(x, y, 'b-', label='f(x)')
plt.plot(xmin, fmin, 'ro', label='Minimum')
plt.title('Golden-Section Search using goldmin_error')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
