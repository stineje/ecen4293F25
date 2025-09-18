import matplotlib.pyplot as plt
import numpy as np
import time

a = [i for i in range(10000)]
b = [i for i in range(10000)]

# Python loop dot product
tic = time.perf_counter()
dot = 0.0
for i in range(len(a)):
    dot += a[i] * b[i]
toc = time.perf_counter()

print("dot_product = " + str(dot))
print("Computation time = " + str(1000*(toc - tic)) + " ms")

# NumPy dot product
n_tic = time.perf_counter()
n_dot_product = np.array(a).dot(np.array(b))
n_toc = time.perf_counter()

print("\nn_dot_product = " + str(n_dot_product))
print("Computation time = " + str(1000*(n_toc - n_tic)) + " ms")
