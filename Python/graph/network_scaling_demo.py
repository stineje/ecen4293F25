import networkx as nx
import time
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------
# 1. Graph generator
# ------------------------------------------
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

# ------------------------------------------
# 2. Benchmarking
# ------------------------------------------
def measure_shortest_path_scaling(generator="erdos_renyi", p=0.05):
    """Measure time to compute shortest path as n grows."""
    sizes = [100, 200, 400, 800, 1600]
    times = []
    sample_graphs = []   # keep small graphs for visualization

    for n in sizes:
        G = generate_graph(n, p, generator)
        if n <= 400:
            sample_graphs.append((n, G))

        source = 0
        target = n - 1 if generator != "grid" else (0, (int(np.sqrt(n)) - 1, int(np.sqrt(n)) - 1))

        start = time.time()
        try:
            nx.shortest_path(G, source=source, target=target)
        except nx.NetworkXNoPath:
            pass
        end = time.time()

        t = end - start
        times.append(t)
        print(f"n={n:4d}  time={t:8.6f} s")

    return np.array(sizes), np.array(times), sample_graphs

# ------------------------------------------
# 3. Plot scaling and save PNG
# ------------------------------------------
def plot_scaling(sizes, times):
    """Plot runtime scaling vs. theoretical O(n²) and O(n)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(sizes, times, 'ro-', label='Measured runtime')
    ax.loglog(sizes, 1e-7 * sizes**2, 'b--', label=r'O($n^2$) reference')
    ax.loglog(sizes, 5e-6 * sizes, 'g-.', label=r'O($n$) reference')
    ax.set_xlabel('Number of Nodes (n)')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Shortest Path Scaling in NetworkX')
    ax.legend()
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()

    filename = "networkx_scaling.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    if os.path.exists(filename):
        print(f"Scaling plot saved to {filename}")
    else:
        print("Warning: PNG save failed!")

    plt.show()

# ------------------------------------------
# 4. Plot example graphs and save PNG
# ------------------------------------------
def plot_example_graphs(sample_graphs, generator):
    """Show representative graphs for visualization."""
    fig, axes = plt.subplots(1, len(sample_graphs), figsize=(15, 4))
    if len(sample_graphs) == 1:
        axes = [axes]  # handle single graph case

    for ax, (n, G) in zip(axes, sample_graphs):
        pos = nx.spring_layout(G, seed=0)
        nx.draw(G, pos, node_color='skyblue', edge_color='gray',
                node_size=100, with_labels=False, ax=ax)
        ax.set_title(f"{generator}\n(n={n})")
        ax.axis('off')

    plt.suptitle(f"Example Graphs from {generator} Generator", fontsize=14)
    plt.tight_layout()

    filename = f"{generator}_examples.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    if os.path.exists(filename):
        print(f"Example graphs saved to {filename}")
    else:
        print("Warning: PNG save failed!")

    plt.show()

# ------------------------------------------
# 5. Main
# ------------------------------------------
if __name__ == "__main__":
    generator_type = "barabasi_albert"  # try "erdos_renyi", "watts_strogatz", or "grid"
    sizes, times, sample_graphs = measure_shortest_path_scaling(generator=generator_type)
    plot_scaling(sizes, times)
    plot_example_graphs(sample_graphs, generator_type)
