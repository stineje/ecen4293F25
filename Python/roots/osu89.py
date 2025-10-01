from bisect1 import bisect1
import matplotlib.pyplot as plt
import numpy as np


def f(xp):
    return np.log(xp**2)-0.3


(xp, fm, ea, iter) = bisect1(f, -2.5, 0.5)
print(f(xp))
print('xp = {0:7.8f}'.format(xp))
print('function value = {0:7.3g}'.format(fm))
print('relative error = {0:7.3g}'.format(ea))
print('iterations = {0:5d}'.format(iter))
