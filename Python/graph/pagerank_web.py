import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# --------------------------
# 1) Graph definition
# --------------------------
nodes = ["Google", "Facebook", "YouTube", "Twitter", "Wikipedia", "Amazon"]

# Raw hyperlinks (no teleport yet). You can tweak these edges freely.
# Each tuple is (source, target). Multiple targets from the same source indicate outgoing links.
edges = [
    ("Google", "YouTube"),
    ("Google", "Wikipedia"),
    ("Google", "Twitter"),

    ("Facebook", "Google"),
    ("Facebook", "YouTube"),

    ("YouTube", "Google"),
    ("YouTube", "Twitter"),
    ("YouTube", "Wikipedia"),

    ("Twitter", "YouTube"),
    ("Twitter", "Wikipedia"),
    ("Twitter", "Amazon"),

    ("Wikipedia", "Google"),
    ("Wikipedia", "YouTube"),
    ("Wikipedia", "Amazon"),

    ("Amazon", "Google"),
    ("Amazon", "Wikipedia"),
]

N = len(nodes)
node_index = {n: i for i, n in enumerate(nodes)}

# --------------------------
# 2) Row-stochastic transition (link) matrix P_raw
# --------------------------
P_raw = np.zeros((N, N), dtype=float)
out_counts = defaultdict(int)
for u, v in edges:
    out_counts[u] += 1
for u, v in edges:
    i, j = node_index[u], node_index[v]
    P_raw[i, j] += 1.0 / out_counts[u]

# Handle dangling nodes (no outlinks): make them uniform
for n in nodes:
    i = node_index[n]
    if out_counts[n] == 0:
        P_raw[i, :] = 1.0 / N

# --------------------------
# 3) Google (Markov) matrix with damping/teleport
# --------------------------
alpha = 0.85                     # damping (probability to follow a link)
teleport = (1 - alpha) / N       # uniform teleport probability
G = alpha * P_raw + teleport * np.ones((N, N))

# Sanity check: rows should sum to 1
assert np.allclose(G.sum(axis=1), 1.0, atol=1e-12)

# --------------------------
# 4) Stationary distribution (Markov “rate”) via power iteration
# --------------------------
def power_iteration(M, tol=1e-12, max_iter=10_000):
    pi = np.ones(M.shape[0]) / M.shape[0]   # start uniform
    for _ in range(max_iter):
        new_pi = pi @ M                     # left eigenvector: row vector times M
        if np.linalg.norm(new_pi - pi, 1) < tol:
            return new_pi / new_pi.sum()
        pi = new_pi
    return pi / pi.sum()

pi_power = power_iteration(G)

# Cross-check with NetworkX PageRank (uses right-eigenvector internally with damping=alpha)
Gnx = nx.DiGraph()
Gnx.add_nodes_from(nodes)
Gnx.add_edges_from(edges)
pr_nx = nx.pagerank(Gnx, alpha=alpha)  # returns dict

# Put nx result in same order as 'nodes'
pi_nx = np.array([pr_nx[n] for n in nodes])

# --------------------------
# 5) k-step transition from a starting distribution (optional demo)
# --------------------------
v0 = np.zeros(N); v0[node_index["Google"]] = 1.0  # start at "Google"
k = 3
vk = v0 @ np.linalg.matrix_power(G, k)

# --------------------------
# 6) Print results
# --------------------------
print("\n--- Stationary distribution (power iteration) ---")
for n, p in zip(nodes, pi_power):
    print(f"{n:10s}: {p:.6f}")

print("\n--- Stationary distribution (networkx.pagerank) ---")
for n, p in zip(nodes, pi_nx):
    print(f"{n:10s}: {p:.6f}")

print("\nL1 difference between methods:", np.linalg.norm(pi_power - pi_nx, 1))

print(f"\nDistribution after {k} steps starting at Google:")
for n, p in zip(nodes, vk):
    print(f"{n:10s}: {p:.6f}")

# --------------------------
# 7) Plot and save
# --------------------------
# We'll label only the *real* hyperlinks with their effective one-step probability under G:
#   G_ij = alpha * (1/outdeg_i) + teleport
# Teleport edges (dense) are not drawn to avoid clutter.

edge_labels = {}
for u, v in edges:
    i, j = node_index[u], node_index[v]
    # probability to go u->v in the *Google matrix* along a linked edge
    p_eff = alpha * (1.0 / out_counts[u]) + teleport
    edge_labels[(u, v)] = f"{p_eff:.3f}"

plt.figure(figsize=(9, 6))
pos = nx.spring_layout(Gnx, seed=7)  # deterministic layout

# Node sizes proportional to stationary probability (nice for teaching)
sizes = 3000 * (pi_power / pi_power.max())

nx.draw_networkx_nodes(Gnx, pos, node_size=sizes, node_color="#DDEEFF", edgecolors="black")
nx.draw_networkx_labels(Gnx, pos, font_size=10)
nx.draw_networkx_edges(Gnx, pos, arrows=True, arrowstyle="-|>", width=1.2)
nx.draw_networkx_edge_labels(Gnx, pos, edge_labels=edge_labels, font_size=8)

plt.title("Web Graph with Effective One-Step Markov Probabilities")
plt.axis("off")

out_png = "pagerank_web.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved plot to: {out_png}")
