import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ----- Graph: nodes and directed edges -----
nodes = ["Google", "Facebook", "YouTube", "Twitter", "Wikipedia", "Amazon"]
edges = [
    ("Google", "YouTube"), ("Google", "Wikipedia"), ("Google", "Twitter"),
    ("Facebook", "Google"), ("Facebook", "YouTube"),
    ("YouTube", "Google"), ("YouTube", "Twitter"), ("YouTube", "Wikipedia"),
    ("Twitter", "YouTube"), ("Twitter", "Wikipedia"), ("Twitter", "Amazon"),
    ("Wikipedia", "Google"), ("Wikipedia", "YouTube"), ("Wikipedia", "Amazon"),
    ("Amazon", "Google"), ("Amazon", "Wikipedia"),
]

N = len(nodes)
idx = {n: i for i, n in enumerate(nodes)}

# ----- Build raw row-stochastic link matrix P_raw -----
P_raw = np.zeros((N, N), dtype=float)
outdeg = {n: 0 for n in nodes}
for u, v in edges:
    outdeg[u] += 1
for u, v in edges:
    i, j = idx[u], idx[v]
    P_raw[i, j] += 1.0 / outdeg[u]  # each outlink equally likely

# Handle possible dangling nodes (none here; kept for robustness)
for n in nodes:
    if outdeg[n] == 0:
        P_raw[idx[n], :] = 1.0 / N

# ----- Google/Markov matrix with damping (teleport) -----
alpha = 0.85
G = alpha * P_raw + (1 - alpha) / N * np.ones((N, N))
assert np.allclose(G.sum(axis=1), 1.0)  # rows sum to 1

# ----- Power iteration & history (start from a one-hot for visible transients) -----
T = 20                           # number of iterations to plot
pi = np.zeros(N); pi[idx["Google"]] = 1.0   # start at Google
history = [pi.copy()]
for _ in range(T):
    pi = pi @ G                  # left-eigenvector iteration
    history.append(pi.copy())
history = np.vstack(history)     # shape (T+1, N)

# ----- (Optional) stationary check with networkx -----
Gnx = nx.DiGraph()
Gnx.add_nodes_from(nodes)
Gnx.add_edges_from(edges)
pr = nx.pagerank(Gnx, alpha=alpha)
pi_nx = np.array([pr[n] for n in nodes])
# print("Stationary (nx):", dict(zip(nodes, np.round(pi_nx, 6))))

# ----- Plot -----
plt.figure(figsize=(5.2, 3.6))
for j, name in enumerate(nodes):
    plt.plot(history[:, j], label=name)
plt.xlabel("Iteration")
plt.ylabel("Probability")
plt.ylim(0.0, 0.4)         
plt.grid(True)
plt.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
plt.tight_layout()

# Save and show
out_path = "pagerank_web.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved plot to {out_path}")
