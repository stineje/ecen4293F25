
import matplotlib.pyplot as plt
import numpy as np
import math 

# Single Precision
eps_single = np.finfo(np.float32).eps
print("Single precision machine eps = {}".format(eps_single))

# Double Precision
eps_double = np.finfo(np.float64).eps
print("Double precision machine eps = {}".format(eps_double))
