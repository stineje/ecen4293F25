from bisect1 import bisect1
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**2-9


mp = np.linspace(0.0, 10.0)
plt.plot(mp, f(mp), c='k', lw=0.5)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
