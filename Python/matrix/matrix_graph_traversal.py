import numpy as np

# adjacency matrix (A[i,j] = 1 means i → j)
A = np.array([
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
], dtype=float)

start = 0
n = A.shape[0]

frontier = np.zeros(n)
frontier[start] = 1        # Start at node 0
visited = frontier.copy()

level = 0
print(f"Level {level}: {np.where(frontier>0)[0]}")

while True:
    # Move frontier one step using matrix multiplication
    next_frontier = A.T @ frontier

    # Remove nodes already visited
    next_frontier *= (visited == 0)

    if not next_frontier.any():
        break

    visited += (next_frontier > 0).astype(float)
    level += 1
    print(f"Level {level}: {np.where(next_frontier>0)[0]}")

    frontier = (next_frontier > 0).astype(float)
    
