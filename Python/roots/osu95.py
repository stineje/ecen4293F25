from newtraph import newtraph
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**2-9 - 4.2


def g(x):
    return 2*x


(xsoln, fxsoln, ea, n) = newtraph(f, g, 1, Ea=1.0e-5)
print('Solution = {0:8.15g}'.format(xsoln))
print('Function value at solution = {0:8.15g}'.format(fxsoln))
print('Relative error = {0:8.3e}'.format(ea))
print('Number of iterations = {0:5d}'.format(n))
