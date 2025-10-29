import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Generate graphs using built-in generators
# ---------------------------
def generate_graph(n, p=0.05, generator="erdos_renyi"):
    """Return a NetworkX graph with n nodes using the chosen generator."""
    if generator == "erdos_renyi":
        return nx.erdos_renyi_graph(n, p)
    elif generator == "barabasi_albert":
        return nx.barabasi_albert_graph(n, max(2, int(n * p)))
    elif generator == "watts_strogatz":
        return nx.watts_strogatz_graph(n, k=max(4, int(n * p)), p=0.3)
    elif generator == "grid":
        size = int(np.sqrt(n))
        return nx.grid_2d_graph(size, size)
    else:
        raise ValueError(f"Unknown generator type: {generator}")

# ---------------------------
# 2. Benchmark function
# ---------------------------
def measure_shortest_path_scaling(generator="erdos_renyi", p=0.05, max_nodes=2000):
    """Measure time to compute shortest path as n grows."""
    sizes = [100, 200, 400, 800, 1600]
    times = []

    for n in sizes:
        G = generate_graph(n, p, generator)
        source, target = 0, n - 1 if generator != "grid" else (0, (int(np.sqrt(n)) - 1, int(np.sqrt(n)) - 1))

        start = time.time()
        try:
            nx.shortest_path(G, source=source, target=target)
        except nx.NetworkXNoPath:
            pass
        end = time.time()

        t = end - start
        times.append(t)
        print(f"n={n:4d}  time={t:8.6f} s")

    return np.array(sizes), np.array(times)

# ---------------------------
# 3. Plot scaling
# ---------------------------
def plot_scaling(sizes, times):
    """Plot runtime scaling vs. theoretical O(n^2)."""
    plt.figure(figsize=(7, 5))
    plt.loglog(sizes, times, 'ro-', label='Measured runtime')
    plt.loglog(sizes, 1e-7 * sizes**2, 'b--', label=r'O($n^2$) reference')
    plt.loglog(sizes, 5e-6 * sizes, 'g-.', label=r'O($n$) reference')
    plt.xlabel('Number of Nodes (n)')
    plt.ylabel('Runtime (s)')
    plt.title('Shortest Path Scaling in NetworkX')
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig("networkx_scaling.png", dpi=300)
    print("Plot saved to networkx_scaling.png")
    plt.show()

# ---------------------------
# 4. Main script
# ---------------------------
if __name__ == "__main__":
    generator_type = "erdos_renyi"   # Try: "barabasi_albert", "watts_strogatz", "grid"
    sizes, times = measure_shortest_path_scaling(generator=generator_type)
    plot_scaling(sizes, times)
