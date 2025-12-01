import numpy as np
import time

# -----------------------------------------
# Matrix Generators
# -----------------------------------------

def random_matrix(n, m):
    return np.random.randn(n, m)

def nearly_dependent_matrix(n, m):
    A = np.random.randn(n, m)
    if m > 1:
        A[:, -1] = A[:, 0] * 0.999999 + 1e-6
    return A

def tall_skinny():
    return np.random.randn(200, 10)

def wide_matrix():
    return np.random.randn(10, 200)

def hilbert_like(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1.0)
    return H

def sparse_banded(n, bandwidth=3):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-bandwidth), min(n, i+bandwidth+1)):
            A[i, j] = np.random.randn()
    return A

# -----------------------------------------
# Error Metrics
# -----------------------------------------

def accuracy(A, Q, R):
    """Reconstruction error ||A - QR||_2."""
    return np.linalg.norm(A - Q @ R)

def orthogonality(Q):
    """Deviation from orthogonality ||Q^T Q - I||_2."""
    n = Q.shape[1]
    return np.linalg.norm(Q.T @ Q - np.eye(n))

def time_qr(qr_func, A):
    """Time a QR routine using perf_counter."""
    start = time.perf_counter()
    Q, R = qr_func(A)
    t = time.perf_counter() - start
    return Q, R, t

# -----------------------------------------
# SciPy-based QR "solution"
# -----------------------------------------

def scipy_qr(A):
    """
    Wrapper around SciPy's Householder QR.
    This is the 'golden' stable implementation.
    """
    import scipy.linalg as la
    # mode='economic' gives m x n Q and n x n R for m >= n
    Q, R = la.qr(A, mode='economic', pivoting=False)
    return Q, R

# Optional: reference using NumPy as well
def numpy_qr(A):
    Q, R = np.linalg.qr(A, mode='reduced')
    return Q, R

# -----------------------------------------
# Example Driver Comparing SciPy vs NumPy
# -----------------------------------------

def experiment():
    np.set_printoptions(precision=3, suppress=True)

    tests = [
        ("Random 200x200",      random_matrix(200, 200)),
        ("Nearly Dep 200x20",   nearly_dependent_matrix(200, 20)),
        ("Tall-skinny 200x10",  tall_skinny()),
        ("Wide 10x200",         wide_matrix()),
        ("Hilbert-like 40x40",  hilbert_like(40)),
        ("Sparse banded 200x200", sparse_banded(200, bandwidth=3)),
    ]

    methods = [
        ("SciPy QR", scipy_qr),
        ("NumPy QR", numpy_qr),
    ]

    for name, A in tests:
        print("\n=== Matrix:", name, "===")
        for mname, mfunc in methods:
            try:
                Q, R, t = time_qr(mfunc, A)
                acc = accuracy(A, Q, R)
                ortho = orthogonality(Q)
                print(f"{mname:10s}  time={t:7.4f}s  "
                      f"||A-QR||={acc:8.2e}  ||Q^TQ-I||={ortho:8.2e}")
            except Exception as e:
                print(f"{mname:10s}  FAILED with error: {e}")

if __name__ == "__main__":
    experiment()
