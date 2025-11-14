import numpy as np
import time
import matplotlib.pyplot as plt

def gemm_naive(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=A.dtype)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C

def time_gemm(kernel, n, dtype=np.float64):
    A = np.random.rand(n, n).astype(dtype)
    B = np.random.rand(n, n).astype(dtype)
    # warm-up
    C = kernel(A, B)
    t0 = time.perf_counter()
    C = kernel(A, B)
    t1 = time.perf_counter()
    dt = t1 - t0
    return dt
