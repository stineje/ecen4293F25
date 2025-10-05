from brentsimp import brentsimp
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**2-9


def g(x):
    return 2*x


xsoln = brentsimp(f, 0, 5)
print('Solution = {0:8.15g}'.format(xsoln))
