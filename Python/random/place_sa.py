import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import random

# -----------------------------
# Toy netlist generator
# -----------------------------


def make_random_netlist(num_cells=30, num_nets=45, net_degree_range=(2, 4), seed=3):
    rng = np.random.default_rng(seed)
    nets = []
    for _ in range(num_nets):
        k = rng.integers(net_degree_range[0], net_degree_range[1] + 1)
        pins = rng.choice(num_cells, size=k, replace=False).tolist()
        nets.append(pins)
    return nets

# HPWL cost (half-perimeter wirelength)


def hpwl(nets, pos):
    wl = 0.0
    for pins in nets:
        xs = pos[pins, 0]
        ys = pos[pins, 1]
        wl += (xs.max() - xs.min()) + (ys.max() - ys.min())
    return wl

# Simple density penalty on a grid of bins


def density_penalty(pos, grid_size, bins=(8, 8), target_util=0.75):
    W, H = grid_size
    bx, by = bins
    # Count cells per bin
    x_idx = np.clip((pos[:, 0] * bx / W).astype(int), 0, bx-1)
    y_idx = np.clip((pos[:, 1] * by / H).astype(int), 0, by-1)
    counts = np.zeros((bx, by), dtype=int)
    for xi, yi in zip(x_idx, y_idx):
        counts[xi, yi] += 1
    # Target per bin:
    target = target_util * (len(pos) / (bx * by))
    # Quadratic overflow penalty
    over = np.maximum(0.0, counts - target)
    return float((over**2).sum())

# Combined objective


def cost(nets, pos, grid_size, lam=0.2):
    return hpwl(nets, pos) + lam * density_penalty(pos, grid_size)

# -----------------------------
# Simulated annealing
# -----------------------------


def accept_prob(e_curr, e_new, T):
    if e_new <= e_curr:
        return 1.0
    return math.exp(-(e_new - e_curr) / max(T, 1e-12))


def schedule_geometric(t, T0=5.0, alpha=0.995, Tmin=1e-4):
    return max(Tmin, T0 * (alpha ** t))


def anneal(nets, pos0, grid_size=(40, 40), iters=1500, step=1.5, lam=0.2, T0=5.0, alpha=0.995):
    rng = np.random.default_rng(7)
    pos = pos0.copy()
    W, H = grid_size
    e = cost(nets, pos, grid_size, lam)
    path = [(pos.copy(), e, schedule_geometric(0, T0, alpha))]
    for t in range(1, iters+1):
        T = schedule_geometric(t, T0, alpha)
        pos_new = pos.copy()
        # Move: either jitter one cell or swap two cells (keeps variety)
        if rng.random() < 0.7:
            i = rng.integers(0, len(pos))
            delta = rng.normal(0, step, size=2)
            pos_new[i] = np.clip(pos_new[i] + delta, [0, 0], [W, H])
        else:
            i, j = rng.choice(len(pos), size=2, replace=False)
            pos_new[i], pos_new[j] = pos_new[j].copy(), pos_new[i].copy()

        e_new = cost(nets, pos_new, grid_size, lam)
        if rng.random() < accept_prob(e, e_new, T):
            pos, e = pos_new, e_new
        path.append((pos.copy(), e, T))
    return path


# -----------------------------
# Build a toy instance
# -----------------------------
N_CELLS = 40
GRID = (50, 50)   # width, height
NETS = make_random_netlist(
    num_cells=N_CELLS, num_nets=55, net_degree_range=(2, 5), seed=5)

rng = np.random.default_rng(2)
pos0 = np.column_stack([
    rng.uniform(0, GRID[0], size=N_CELLS),
    rng.uniform(0, GRID[1], size=N_CELLS),
])

# Run SA
T0, ALPHA, LAMBDA = 6.0, 0.996, 0.25
PATH = anneal(NETS, pos0, grid_size=GRID, iters=1800,
              step=1.8, lam=LAMBDA, T0=T0, alpha=ALPHA)

# Extract series
costs = np.array([e for (_, e, _) in PATH])
temps = np.array([T for (_, _, T) in PATH])

# -----------------------------
# Visualization: placement + HPWL breakdown + temperature
# -----------------------------
fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], hspace=0.30, wspace=0.22)

axP = fig.add_subplot(gs[0, 0])
axP.set_title("Placement (cells and nets)")
axP.set_xlim(0, GRID[0])
axP.set_ylim(0, GRID[1])
axP.set_xlabel("x")
axP.set_ylabel("y")

axC = fig.add_subplot(gs[0, 1])
axC.set_title("Cost components")
axC.set_xlabel("Iteration")
axC.set_ylabel("Value")
axC.grid(True, alpha=0.25)

axT = fig.add_subplot(gs[1, :])
axT.set_title("Temperature and Total Cost")
axT.set_xlabel("Iteration")
axT.grid(True, alpha=0.25)

# Artists
cell_scatter = axP.scatter([], [], s=35, marker='s')
net_lines = [axP.plot([], [], lw=0.7, alpha=0.5)[0] for _ in NETS]
txtP = axP.text(0.02, 0.98, "", transform=axP.transAxes, va="top")

line_HPWL, = axC.plot([], [], label="HPWL")
line_DEN, = axC.plot([], [], label="DensityPenalty")
axC.legend(loc="upper right")

line_T,    = axT.plot([], [], label="Temperature (T)")
line_COST, = axT.plot([], [], label="Total Cost")
axT.legend(loc="upper right")

# Helpers to recompute components per frame quickly


def hpwl_quick(pos):
    return hpwl(NETS, pos)


def density_quick(pos):
    return density_penalty(pos, GRID, bins=(8, 8), target_util=0.8)


HPWL_vals = []
DEN_vals = []


def init():
    cell_scatter.set_offsets(np.empty((0, 2)))
    for l in net_lines:
        l.set_data([], [])
    line_HPWL.set_data([], [])
    line_DEN.set_data([], [])
    line_T.set_data([], [])
    line_COST.set_data([], [])
    txtP.set_text("")
    return [cell_scatter, *net_lines, line_HPWL, line_DEN, line_T, line_COST, txtP]


def update(frame):
    pos, e, T = PATH[frame]
    # Update placement plot
    cell_scatter.set_offsets(pos)

    # Draw nets as polylines between pin bbox corners (approx: connect to centroid)
    for ln, pins in zip(net_lines, NETS):
        xs = pos[pins, 0]
        ys = pos[pins, 1]
        # Plot as a star-to-centroid to reduce clutter
        cx, cy = xs.mean(), ys.mean()
        xs_plot = np.column_stack([xs, np.full_like(xs, cx)]).ravel(order='F')
        ys_plot = np.column_stack([ys, np.full_like(ys, cy)]).ravel(order='F')
        ln.set_data(xs_plot, ys_plot)

    # Recompute components (for live lines)
    hp = hpwl_quick(pos)
    dn = density_quick(pos)
    HPWL_vals.append(hp)
    DEN_vals.append(dn)

    its = np.arange(frame+1)
    line_HPWL.set_data(its, HPWL_vals)
    line_DEN.set_data(its,  DEN_vals)

    line_T.set_data(its, temps[:frame+1])
    line_COST.set_data(its, costs[:frame+1])

    # Autoscale y for line plots during early frames
    if frame % 30 == 0 or frame < 30:
        # Cost components panel
        y_min = min(min(HPWL_vals), min(DEN_vals))
        y_max = max(max(HPWL_vals), max(DEN_vals))
        pad = 0.06 * (y_max - y_min + 1e-9)
        axC.set_xlim(0, len(PATH)-1)
        axC.set_ylim(y_min - pad, y_max + pad)
        # Temp/cost panel
        y2_min = min(temps[:frame+1].min(), costs[:frame+1].min())
        y2_max = max(temps[:frame+1].max(), costs[:frame+1].max())
        pad2 = 0.06 * (y2_max - y2_min + 1e-9)
        axT.set_xlim(0, len(PATH)-1)
        axT.set_ylim(y2_min - pad2, y2_max + pad2)

    txtP.set_text(
        f"iter: {frame}\nT: {T:.4f}\nTotal Cost: {e:.1f}\nHPWL: {hp:.1f}  Den: {dn:.1f}")
    return [cell_scatter, *net_lines, line_HPWL, line_DEN, line_T, line_COST, txtP]


anim = FuncAnimation(fig, update, frames=len(PATH), init_func=init,
                     interval=10, blit=True, repeat=False)

plt.tight_layout()
plt.show()

# Optional save (uncomment one). Requires pillow for GIF, ffmpeg for MP4.
# anim.save("sa_vlsi_placement.gif", dpi=130, fps=60)
# anim.save("sa_vlsi_placement.mp4", dpi=130, fps=60)
