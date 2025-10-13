from goldmin import goldmin
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**2/10 - 2*np.sin(x)


xl = 0
xu = 4
xmin, fmin, ea, n = goldmin(f, xl, xu, Ea=1.0e-5)
print('Solution = {0:8.15g}'.format(xmin))
print('Function value at solution = {0:8.15g}'.format(fmin))
print('Relative error = {0:8.3e}'.format(ea))
print('Number of iterations = {0:5d}'.format(n))
