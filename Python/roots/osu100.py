from brentsimp import brentsimp
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**10 - 1


xsoln = brentq(f, 0, 1, rtol=1.0e-7, xtol=1.0e-7,
               maxiter=20, full_output=True)
print(xsoln)

