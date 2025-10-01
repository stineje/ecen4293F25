from regfal import regfal
import matplotlib.pyplot as plt
import numpy as np


def f(m):
    g = 9.81
    cd = 0.25
    t = 4
    v = 36
    return np.sqrt(m*g/cd)*np.tanh(np.sqrt(g*cd/m)*t)-v


(m, fm, ea, iter) = regfal(f, 50, 200)
print('mass = {0:7.8f} kg'.format(m))
print('function value = {0:7.3g}'.format(fm))
print('relative error = {0:7.3g}'.format(ea))
print('iterations = {0:5d}'.format(iter))
