from book import bisect
import matplotlib.pyplot as plt
import numpy as np


def f(cd):
    g = 9.81
    m = 95
    t = 9
    v = 46
    return np.sqrt(m*g/cd)*np.tanh(np.sqrt(g*cd/m)*t)-v


(cd, Ea, ea, iter) = bisect(f, 0.2, 0.5, es=1.e-7)
print('drag coefficient = {0:10.6f} m/kg'.format(cd))
print('absolute error = {0:7.3g}'.format(Ea))
print('relative error = {0:7.3g}'.format(ea))
print('iterations = {0:5d}'.format(iter))
