from fixpt import fixpt
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.exp(-x)


(Hsoln, ea, n) = fixpt(f, 0, Ea=1.e-8)
print('Solution = {0:8.15g}'.format(Hsoln))
print('Relative error = {0:8.3e}'.format(ea))
print('Number of iterations = {0:5d}'.format(n))
