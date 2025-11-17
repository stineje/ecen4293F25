import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Directed graph: 0 → 1, 0 → 2, 2 → 3
G = nx.DiGraph()
G.add_edges_from([(0, 1), (0, 2), (2, 3)])

# Nice fixed layout for slides
pos = {0: (0, 0.5), 1: (1, 1), 2: (1, 0), 3: (2, 0.5)}

plt.figure()
nx.draw(G, pos, with_labels=True, arrows=True)
plt.title("Example Graph for Matrix-Based Traversal")
plt.axis("off")
plt.savefig("matrix_graph_traversal_networkx.png")
plt.show()

# Adjacency matrix A (as NumPy array)
A = nx.to_numpy_array(G, nodelist=[0, 1, 2, 3], dtype=float)
print("Adjacency matrix A:\n", A)

# Start BFS-like traversal from node 0 using matrix–vector multiplies
n = A.shape[0]
frontier = np.zeros(n)
frontier[0] = 1.0       # start at node 0
visited = frontier.copy()

level = 0
print(f"Level {level}: nodes {np.where(frontier > 0)[0]}")

while True:
    # Move one step: multiply by A^T
    next_frontier = A.T @ frontier

    # Remove already visited nodes
    next_frontier *= (visited == 0)

    if not next_frontier.any():
        break

    visited += (next_frontier > 0).astype(float)
    level += 1
    print(f"Level {level}: nodes {np.where(next_frontier > 0)[0]}")

    frontier = (next_frontier > 0).astype(float)
