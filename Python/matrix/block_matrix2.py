import numpy as np

def block_multiply(A, B, block_size):
    """Function to perform block matrix multiplication."""
    n = A.shape[0]
    C = np.zeros((n, n), dtype=int)

    # Basic validity check
    if n % block_size != 0:
        raise ValueError("Matrix size N must be divisible by block_size.")

    # Perform block multiplication
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                A_block = A[i:i+block_size, k:k+block_size]
                B_block = B[k:k+block_size, j:j+block_size]
                C[i:i+block_size, j:j+block_size] += np.dot(A_block, B_block)

    return C


# ======== USER-SPECIFIED MATRIX SIZE ========
N = 128            # <-- Set any size you want: 8, 12, 64, 128, etc.
block_size = 16    # <-- Must divide N evenly

# Generate random N×N matrices A and B
A = np.random.randint(0, 10, (N, N))
B = np.random.randint(0, 10, (N, N))

# Perform block multiplication
C = block_multiply(A, B, block_size)

print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

print("\nResult of Block Matrix Multiplication (A * B):")
print(C)

# Verify correctness
C_direct = A @ B   # same as np.dot(A, B)
print("\nDirect Multiplication Result (A * B):")
print(C_direct)

print("Does block multiplication match direct multiplication?")
print(np.array_equal(C, C_direct))
