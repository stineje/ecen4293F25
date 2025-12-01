import numpy as np
import time

# -------------------------
# Matrix Generators
# -------------------------

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

# -------------------------
# Error Measures
# -------------------------

def accuracy(A, Q, R):
    return np.linalg.norm(A - Q @ R)

def orthogonality(Q):
    n = Q.shape[1]
    return np.linalg.norm(Q.T @ Q - np.eye(n))

def time_qr(qr_func, A):
    start = time.perf_counter()
    Q, R = qr_func(A)
    t = time.perf_counter() - start
    return Q, R, t

# -------------------------
# QR Algorithms
# -------------------------

def qr_cgs(A):
    """Classical Gram–Schmidt QR."""
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 0:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = 0
    return Q, R

def qr_mgs(A):
    """Modified Gram–Schmidt QR."""
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy()
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        if R[i, i] > 0:
            Q[:, i] = V[:, i] / R[i, i]
        else:
            Q[:, i] = 0
        for j in range(i+1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]
    return Q, R

def qr_householder(A):
    """Householder QR."""
    A = np.array(A, dtype=float)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(min(m, n)):
        x = R[k:, k]
        normx = np.linalg.norm(x)
        if normx == 0:
            continue
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        v = x + np.sign(x[0]) * normx * e1
        v = v / np.linalg.norm(v)
        Hk = np.eye(m)
        Hk_sub = np.eye(len(x)) - 2.0 * np.outer(v, v)
        Hk[k:, k:] = Hk_sub
        R = Hk @ R
        Q = Q @ Hk.T
    # Trim Q,R to economic size
    return Q, R

def qr_givens(A):
    """Givens rotation QR (simple dense implementation, not optimized)."""
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for j in range(n):
        for i in range(m-1, j, -1):
            if abs(R[i, j]) > 1e-14:
                r = np.hypot(R[i-1, j], R[i, j])
                if r == 0:
                    continue
                c = R[i-1, j] / r
                s = -R[i, j] / r
                G = np.eye(m)
                G[[i-1, i], [i-1, i]] = c
                G[i, i-1] = s
                G[i-1, i] = -s
                R = G @ R
                Q = Q @ G.T
    return Q, R

def experiment():
    np.set_printoptions(precision=3, suppress=True)

    tests = [
        ("Random 200x200",      random_matrix(200, 200)),
        ("Nearly Dep 200x20",   nearly_dependent_matrix(200, 20)),
        ("Tall-skinny 200x10",  tall_skinny()),
        ("Wide 10x200",         wide_matrix()),
        ("Hilbert-like 20x20",  hilbert_like(20)),
        ("Sparse banded 200x200", sparse_banded(200, bandwidth=3))
    ]

    methods = [
        ("CGS", qr_cgs),
        ("MGS", qr_mgs),
        ("HH", qr_householder),
        ("Givens", qr_givens),
    ]

    for name, A in tests:
        print("\n=== Matrix:", name, "===")
        for mname, mfunc in methods:
            try:
                Q, R, t = time_qr(mfunc, A)
                acc = accuracy(A, Q, R)
                ortho = orthogonality(Q)
                print(f"{mname:12s}  time={t:7.4f}s  ||A-QR||={acc:8.2e}  ||Q^TQ-I||={ortho:8.2e}")
            except Exception as e:
                print(f"{mname:12s}  FAILED with error: {e}")

if __name__ == "__main__":
    experiment()
    
