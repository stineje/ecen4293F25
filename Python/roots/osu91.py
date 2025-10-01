from incsearch import incsearch
import matplotlib.pyplot as plt
import numpy as np

(nb, xb) = incsearch(lambda x: np.sin(10*x)+np.cos(3*x), 3.0, 6.0)

brackets_converted = [(float(a), float(b)) for (a, b) in xb]

for x in brackets_converted:
    print(x)
