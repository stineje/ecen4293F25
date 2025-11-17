import numpy as np
from scipy.sparse import csr_matrix

rows = np.array([0, 0, 1, 2])
cols = np.array([1, 2, 2, 3])
data = np.ones(len(rows))

A_sparse = csr_matrix((data, (rows, cols)), shape=(4, 4))

frontier = np.zeros(4)
frontier[0] = 1.0

for k in range(3):
    frontier = A_sparse.T @ frontier
    print(f"Step {k+1}:", frontier)
