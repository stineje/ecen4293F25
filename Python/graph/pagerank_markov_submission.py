"""
PageRank Algorithm using Markov Matrix Implementation
ECEN 4293: Applied Numerical Methods with Python for Engineers

Author: Student Submission
Date: November 3, 2025

This implementation demonstrates PageRank using Markov chain theory.
PageRank models web surfing as a random walk on a directed graph where:
- Each webpage is a state in a Markov chain
- Links between pages define transition probabilities
- The steady-state distribution gives the PageRank scores
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class PageRankMarkov:
    """
    PageRank implementation using Markov chain approach.

    The PageRank algorithm computes the stationary distribution of a
    random walk on a directed graph. The transition matrix P is constructed
    from the adjacency matrix with damping factor alpha.

    P = (1-alpha)/n * 1 + alpha * M

    where M is the normalized adjacency matrix (column-stochastic).
    """

    def __init__(self, alpha=0.85):
        """
        Initialize PageRank calculator.

        Args:
            alpha (float): Damping factor (probability of following a link).
                          Typical value is 0.85 (Brin & Page, 1998)
        """
        self.alpha = alpha
        self.transition_matrix = None
        self.pagerank_history = []

    def build_transition_matrix(self, adjacency_matrix):
        """
        Build the Google Matrix (transition matrix) from adjacency matrix.

        The transition matrix is column-stochastic, meaning each column sums to 1.
        This represents the probability of transitioning FROM a state.

        Args:
            adjacency_matrix (np.ndarray): Adjacency matrix where A[i,j] = 1
                                          if there's a link from i to j

        Returns:
            np.ndarray: Transition matrix for the Markov chain
        """
        A = adjacency_matrix.astype(float)
        n = A.shape[0]

        # Normalize each column (out-degree normalization)
        # Each column represents where you can go FROM that node
        col_sums = A.sum(axis=0)

        # Handle dangling nodes (nodes with no outgoing links)
        for i in range(n):
            if col_sums[i] == 0:
                # If no outgoing links, distribute probability uniformly
                A[:, i] = 1.0 / n
            else:
                A[:, i] /= col_sums[i]

        # Apply damping factor: Google Matrix
        # P = (1-alpha)/n * E + alpha * M
        # where E is matrix of all 1/n (teleportation)
        E = np.ones((n, n)) / n
        P = (1 - self.alpha) * E + self.alpha * A

        self.transition_matrix = P
        return P

    def compute_pagerank_iterative(self, adjacency_matrix, max_iterations=100,
                                   tolerance=1e-6, initial_rank=None):
        """
        Compute PageRank using iterative method (power iteration).

        This method computes: r^(k+1) = P @ r^(k)
        where P is the transition matrix and r is the rank vector.

        Args:
            adjacency_matrix (np.ndarray): Adjacency matrix of the graph
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance (L1 norm)
            initial_rank (np.ndarray): Initial rank distribution

        Returns:
            tuple: (final_ranks, convergence_history)
        """
        P = self.build_transition_matrix(adjacency_matrix)
        n = P.shape[0]

        # Initialize rank vector (uniform distribution)
        if initial_rank is None:
            rank = np.ones(n) / n
        else:
            rank = initial_rank / np.sum(initial_rank)  # Normalize

        self.pagerank_history = [rank.copy()]

        # Power iteration method
        for iteration in range(max_iterations):
            prev_rank = rank.copy()

            # Matrix-vector multiplication: r^(k+1) = P @ r^(k)
            rank = P @ rank

            # Normalize to maintain probability distribution
            rank = rank / np.sum(rank)

            self.pagerank_history.append(rank.copy())

            # Check convergence using L1 norm
            diff = np.linalg.norm(rank - prev_rank, ord=1)

            if diff < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                print(f"Final L1 difference: {diff:.2e}")
                break
        else:
            print(f"Maximum iterations ({max_iterations}) reached")

        return rank, np.array(self.pagerank_history)

    def compute_pagerank_matrix_power(self, adjacency_matrix, num_iterations=50):
        """
        Compute PageRank using matrix power method.

        This method computes: r^(k) = P^k @ r^(0)
        by explicitly computing powers of the transition matrix.

        Args:
            adjacency_matrix (np.ndarray): Adjacency matrix of the graph
            num_iterations (int): Number of iterations to compute

        Returns:
            tuple: (final_ranks, rank_history)
        """
        P = self.build_transition_matrix(adjacency_matrix)
        n = P.shape[0]

        # Initial uniform distribution
        rank = np.ones(n) / n
        rank_history = [rank.copy()]

        # Compute successive matrix powers
        P_power = P.copy()
        for k in range(1, num_iterations + 1):
            # Compute P^k @ r^(0)
            rank = P_power @ (np.ones(n) / n)
            rank_history.append(rank.copy())

            # Update P_power for next iteration
            if k < num_iterations:
                P_power = P_power @ P

        return rank, np.array(rank_history)

    def get_transition_matrix(self):
        """Return the computed transition matrix."""
        return self.transition_matrix


def create_example_graph():
    """
    Create an example directed graph for demonstration.

    This graph represents a small web with 6 pages and various links.
    Graph structure matches slide 27, L22 from course materials.

    Returns:
        tuple: (NetworkX DiGraph, adjacency matrix as numpy array)
    """
    G = nx.DiGraph()

    # Define edges (links between pages)
    edges = [
        (0, 1), (1, 2), (1, 3), (2, 3), (3, 0),
        (2, 4), (4, 5), (2, 5), (0, 5), (5, 0)
    ]

    G.add_edges_from(edges)

    # Convert to adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()
    adj_matrix = np.array(adj_matrix, dtype=float)

    return G, adj_matrix


def plot_convergence(rank_history, title="PageRank Convergence Over Time"):
    """
    Plot how PageRank values evolve over iterations.

    Args:
        rank_history (np.ndarray): Array of shape (iterations, nodes)
        title (str): Plot title
    """
    plt.figure(figsize=(12, 7))

    num_nodes = rank_history.shape[1]
    iterations = range(rank_history.shape[0])

    # Plot each node's PageRank evolution
    for node in range(num_nodes):
        plt.plot(iterations, rank_history[:, node],
                label=f'Node {node}', marker='o', markersize=4)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('PageRank Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt


def plot_transition_matrix(P, title="Transition Matrix (Google Matrix)"):
    """
    Visualize the transition matrix as a heatmap.

    Args:
        P (np.ndarray): Transition matrix
        title (str): Plot title
    """
    plt.figure(figsize=(10, 8))

    im = plt.imshow(P, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Transition Probability')

    # Add text annotations
    n = P.shape[0]
    for i in range(n):
        for j in range(n):
            text = plt.text(j, i, f'{P[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.xlabel('From Node', fontsize=12)
    plt.ylabel('To Node', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.tight_layout()

    return plt


def plot_comparison(custom_ranks, nx_ranks, node_labels=None):
    """
    Compare custom PageRank implementation with NetworkX.

    Args:
        custom_ranks (np.ndarray): PageRank scores from custom implementation
        nx_ranks (dict): PageRank scores from NetworkX
        node_labels (list): Optional node labels
    """
    plt.figure(figsize=(12, 6))

    n = len(custom_ranks)
    if node_labels is None:
        node_labels = [f'Node {i}' for i in range(n)]

    x = np.arange(n)
    width = 0.35

    # Sort NetworkX results by node order
    nx_ranks_sorted = [nx_ranks[i] for i in range(n)]

    plt.bar(x - width/2, custom_ranks, width, label='Custom (Markov)', alpha=0.8)
    plt.bar(x + width/2, nx_ranks_sorted, width, label='NetworkX', alpha=0.8)

    plt.xlabel('Node', fontsize=12)
    plt.ylabel('PageRank Score', fontsize=12)
    plt.title('PageRank Comparison: Custom vs NetworkX', fontsize=14, fontweight='bold')
    plt.xticks(x, node_labels)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    return plt


def plot_graph_structure(G, pagerank_scores):
    """
    Visualize the graph structure with node sizes based on PageRank.

    Args:
        G (nx.DiGraph): NetworkX directed graph
        pagerank_scores (np.ndarray): PageRank scores for sizing nodes
    """
    plt.figure(figsize=(12, 8))

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2)

    # Node sizes proportional to PageRank
    node_sizes = [score * 10000 for score in pagerank_scores]

    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                          node_color=pagerank_scores, cmap='viridis',
                          alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          arrows=True, arrowsize=20,
                          connectionstyle='arc3,rad=0.1',
                          alpha=0.6, width=2)

    plt.title('Graph Structure (Node size ~ PageRank)',
             fontsize=14, fontweight='bold')
    plt.axis('off')

    # Add colorbar with ax parameter
    ax = plt.gca()
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=min(pagerank_scores),
                                                 vmax=max(pagerank_scores)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='PageRank Score', shrink=0.8)
    plt.tight_layout()

    return plt


def main():
    """Main execution function."""

    print("=" * 70)
    print("PageRank Algorithm using Markov Chain Implementation")
    print("ECEN 4293: Applied Numerical Methods with Python for Engineers")
    print("=" * 70)
    print()

    # Create example graph
    print("Creating example graph...")
    G, adj_matrix = create_example_graph()
    n_nodes = adj_matrix.shape[0]
    n_edges = G.number_of_edges()
    print(f"Graph created with {n_nodes} nodes and {n_edges} edges")
    print()

    # Display adjacency matrix
    print("Adjacency Matrix:")
    print(adj_matrix)
    print()

    # Initialize PageRank calculator
    alpha = 0.85  # Standard damping factor from Brin & Page
    pr = PageRankMarkov(alpha=alpha)

    print(f"Damping factor (alpha): {alpha}")
    print()

    # Build and display transition matrix
    print("Building transition matrix (Google Matrix)...")
    P = pr.build_transition_matrix(adj_matrix)
    print("Transition Matrix P:")
    print(P)
    print()

    # Verify column-stochastic property
    col_sums = P.sum(axis=0)
    print("Column sums (should all be 1.0):")
    print(col_sums)
    print()

    # Compute PageRank using iterative method
    print("-" * 70)
    print("Method 1: Iterative Power Method")
    print("-" * 70)
    custom_ranks, rank_history = pr.compute_pagerank_iterative(
        adj_matrix, max_iterations=100, tolerance=1e-6
    )
    print()
    print("Final PageRank scores (Custom Implementation):")
    for i, rank in enumerate(custom_ranks):
        print(f"  Node {i}: {rank:.6f}")
    print()

    # Compute PageRank using NetworkX for comparison
    print("-" * 70)
    print("Method 2: NetworkX Built-in PageRank")
    print("-" * 70)
    nx_ranks = nx.pagerank(G, alpha=alpha)
    print("PageRank scores (NetworkX):")
    for node in sorted(nx_ranks.keys()):
        print(f"  Node {node}: {nx_ranks[node]:.6f}")
    print()

    # Compare results
    print("-" * 70)
    print("Comparison of Results")
    print("-" * 70)
    print(f"{'Node':<8} {'Custom':<12} {'NetworkX':<12} {'Difference':<12}")
    print("-" * 70)
    for i in range(n_nodes):
        diff = abs(custom_ranks[i] - nx_ranks[i])
        print(f"{i:<8} {custom_ranks[i]:<12.6f} {nx_ranks[i]:<12.6f} {diff:<12.2e}")
    print()

    # Compute maximum difference
    max_diff = max(abs(custom_ranks[i] - nx_ranks[i]) for i in range(n_nodes))
    print(f"Maximum difference: {max_diff:.2e}")
    print()

    # Generate plots
    print("-" * 70)
    print("Generating Visualizations")
    print("-" * 70)

    # Plot 1: Convergence over time
    print("1. Plotting PageRank convergence...")
    plot_convergence(rank_history,
                    "PageRank Convergence: Iterative Power Method")
    plt.savefig('/home/user/ecen4293F25/Python/graph/pagerank_convergence.png',
                dpi=300, bbox_inches='tight')
    print("   Saved: pagerank_convergence.png")

    # Plot 2: Transition matrix heatmap
    print("2. Plotting transition matrix heatmap...")
    plot_transition_matrix(P,
                          f"Google Matrix (α={alpha})")
    plt.savefig('/home/user/ecen4293F25/Python/graph/transition_matrix.png',
                dpi=300, bbox_inches='tight')
    print("   Saved: transition_matrix.png")

    # Plot 3: Comparison bar chart
    print("3. Plotting comparison with NetworkX...")
    plot_comparison(custom_ranks, nx_ranks)
    plt.savefig('/home/user/ecen4293F25/Python/graph/pagerank_comparison.png',
                dpi=300, bbox_inches='tight')
    print("   Saved: pagerank_comparison.png")

    # Plot 4: Graph structure visualization
    print("4. Plotting graph structure...")
    plot_graph_structure(G, custom_ranks)
    plt.savefig('/home/user/ecen4293F25/Python/graph/graph_structure.png',
                dpi=300, bbox_inches='tight')
    print("   Saved: graph_structure.png")

    print()
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  - Implemented PageRank using Markov chain theory")
    print(f"  - Graph size: {n_nodes} nodes, {n_edges} edges")
    print(f"  - Converged in {len(rank_history)-1} iterations")
    print(f"  - Maximum difference vs NetworkX: {max_diff:.2e}")
    print(f"  - Generated 4 visualization plots")
    print()

    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
