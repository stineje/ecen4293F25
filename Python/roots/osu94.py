from wegstein import wegstein
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return np.exp(-x)


(Hsoln, ea, n) = wegstein(f, 0.40, 0.45, Ea=1.e-5)
print('Solution = {0:8.15g}'.format(Hsoln))
print('Relative error = {0:8.3e}'.format(ea))
print('Number of iterations = {0:5d}'.format(n))
