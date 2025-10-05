from fixpt import fixpt
from wegstein import wegstein
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return -np.log(x)


(H1soln, ea1, n1) = wegstein(f, 0.4, 0.45, Ea=1.e-5)
(H2soln, ea2, n2) = fixpt(f, 0, Ea=1.e-5)
print('Solution1 = {0:8.15g}'.format(H1soln))
print('Relative error1 = {0:8.3e}'.format(ea1))
print('Number of iterations1 = {0:5d}'.format(n1))

print('Solution2 = {0:8.15g}'.format(H2soln))
print('Relative error2 = {0:8.3e}'.format(ea2))
print('Number of iterations2 = {0:5d}'.format(n2))
