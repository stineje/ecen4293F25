# --- Simulated Annealing VLSI Placement: Incremental + Animated Dashboard ---
# - Phase A: free SA from random placement
# - Phase B: incremental re-anneal with locks + obstacle (reheat)
# - Helper: rerun_incremental_from_last(...) to modify unlocked ones & rerun
#
# Requirements: matplotlib, numpy
# Optional (for saving GIF/MP4): pillow (GIF) or ffmpeg (MP4)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import math
import random

# =========================
# Netlist / cost functions
# =========================


def make_random_netlist(num_cells=40, num_nets=55, net_degree_range=(2, 5), seed=5):
    rng = np.random.default_rng(seed)
    nets = []
    for _ in range(num_nets):
        k = rng.integers(net_degree_range[0], net_degree_range[1] + 1)
        pins = rng.choice(num_cells, size=k, replace=False).tolist()
        nets.append(pins)
    return nets


def hpwl(nets, pos):
    wl = 0.0
    for pins in nets:
        xs = pos[pins, 0]
        ys = pos[pins, 1]
        wl += (xs.max() - xs.min()) + (ys.max() - ys.min())
    return wl


def density_penalty(pos, grid_size, bins=(8, 8), target_util=0.75):
    W, H = grid_size
    bx, by = bins
    x_idx = np.clip((pos[:, 0] * bx / W).astype(int), 0, bx-1)
    y_idx = np.clip((pos[:, 1] * by / H).astype(int), 0, by-1)
    counts = np.zeros((bx, by), dtype=int)
    for xi, yi in zip(x_idx, y_idx):
        counts[xi, yi] += 1
    target = target_util * (len(pos) / (bx * by))
    over = np.maximum(0.0, counts - target)
    return float((over**2).sum())


def total_cost(nets, pos, grid_size, lam=0.25):
    return hpwl(nets, pos) + lam * density_penalty(pos, grid_size)

# =========================
# Constraints & utilities
# =========================


def in_obstacle(p, obs):
    x, y = p
    for (x0, y0, x1, y1) in obs:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def project_out(p, obs, W, H):
    x, y = p
    for (x0, y0, x1, y1) in obs:
        if x0 <= x <= x1 and y0 <= y <= y1:
            dx = min(abs(x-x0), abs(x-x1))
            dy = min(abs(y-y0), abs(y-y1))
            if dx < dy:
                x = x0-1e-3 if abs(x-x0) < abs(x-x1) else x1+1e-3
            else:
                y = y0-1e-3 if abs(y-y0) < abs(y-y1) else y1+1e-3
    return np.array([np.clip(x, 0, W), np.clip(y, 0, H)], dtype=float)

# =========================
# Simulated annealing core
# =========================


def accept_prob(e_curr, e_new, T):
    if e_new <= e_curr:
        return 1.0
    return math.exp(-(e_new - e_curr) / max(T, 1e-12))


def schedule_geometric(t, T0=6.0, alpha=0.996, Tmin=1e-4):
    return max(Tmin, T0 * (alpha ** t))


def anneal_with_constraints(
    nets, pos0, grid_size=(50, 50), iters=1000, step=1.5,
    lam=0.25, T0=6.0, alpha=0.996,
    locked=None, obstacles=None, swap_prob=0.3, jitter_prob=0.7, seed=7
):
    rng = np.random.default_rng(seed)
    W, H = grid_size
    pos = pos0.copy()
    N = len(pos)
    locked = np.zeros(N, dtype=bool) if locked is None else locked.astype(bool)
    obstacles = [] if obstacles is None else list(obstacles)

    # Ensure legality
    for i in range(N):
        if in_obstacle(pos[i], obstacles):
            pos[i] = project_out(pos[i], obstacles, W, H)

    e = total_cost(nets, pos, grid_size, lam)
    path, costs, temps = [pos.copy()], [e], [schedule_geometric(0, T0, alpha)]

    for t in range(1, iters+1):
        T = schedule_geometric(t, T0, alpha)
        pos_new = pos.copy()

        if rng.random() < jitter_prob:
            cand = np.where(~locked)[0]
            if cand.size > 0:
                i = rng.choice(cand)
                delta = rng.normal(0, step, size=2)
                p = np.clip(pos_new[i] + delta, [0, 0], [W, H])
                if obstacles:
                    p = project_out(p, obstacles, W, H)
                pos_new[i] = p
        else:
            cand = np.where(~locked)[0]
            if cand.size >= 2:
                i, j = rng.choice(cand, size=2, replace=False)
                pos_new[i], pos_new[j] = pos_new[j].copy(), pos_new[i].copy()

        e_new = total_cost(nets, pos_new, grid_size, lam)
        if rng.random() < accept_prob(e, e_new, T):
            pos, e = pos_new, e_new

        path.append(pos.copy())
        costs.append(e)
        temps.append(T)

    return np.array(path), np.array(costs), np.array(temps)


def incremental_anneal(
    nets, pos_current, grid_size=(50, 50), iters=800, step=1.2, lam=0.25,
    reheat_factor=0.15, alpha=0.997, locked=None, obstacles=None, seed=11
):
    curr_cost = total_cost(nets, pos_current, grid_size, lam)
    T0 = max(0.5, reheat_factor * (1.0 + curr_cost / (len(pos_current) + 1e-9)))
    return anneal_with_constraints(
        nets, pos_current, grid_size, iters, step, lam, T0, alpha,
        locked=locked, obstacles=obstacles, seed=seed
    )

# =========================
# Demo runner + animation
# =========================


def run_demo(n_cells=40, grid=(50, 50), num_nets=55, itA=900, itB=900,
             lam=0.25, T0A=6.0, alphaA=0.996, stepA=1.8,
             reheatB=0.18, alphaB=0.997, stepB=1.2,
             lock_idx=(0, 3, 7, 11), obstaclesB=((20, 20, 30, 30),),
             seed=2, net_seed=5):
    """Returns (fig, anim, globals_dict) so you can reuse PATH/COST/TEMP later."""
    # Build instance
    NETS = make_random_netlist(
        num_cells=n_cells, num_nets=num_nets, seed=net_seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    pos0 = np.column_stack([rng.uniform(0, grid[0], n_cells),
                            rng.uniform(0, grid[1], n_cells)])

    # Phase A: free SA
    PATH_A, COST_A, TEMP_A = anneal_with_constraints(NETS, pos0, grid_size=grid,
                                                     iters=itA, step=stepA,
                                                     lam=lam, T0=T0A, alpha=alphaA, seed=7)

    # Phase B: locks + obstacle + reheat
    LOCKED = np.zeros(n_cells, dtype=bool)
    if lock_idx:
        LOCKED[list(lock_idx)] = True
    OBSTACLES = list(obstaclesB) if obstaclesB else []
    PATH_B, COST_B, TEMP_B = incremental_anneal(NETS, PATH_A[-1], grid_size=grid,
                                                iters=itB, step=stepB, lam=lam,
                                                reheat_factor=reheatB, alpha=alphaB,
                                                locked=LOCKED, obstacles=OBSTACLES, seed=11)

    # Merge phases
    PATH = np.vstack([PATH_A, PATH_B])
    COST = np.concatenate([COST_A, COST_B])
    TEMP = np.concatenate([TEMP_A, TEMP_B])
    PHASE_SPLIT = len(PATH_A) - 1

    # --- Plotting ---
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[
                          2.0, 1.0], hspace=0.30, wspace=0.22)

    axP = fig.add_subplot(gs[0, 0])
    axP.set_title("Placement (cells, nets, obstacle, locked)")
    axP.set_xlim(0, grid[0])
    axP.set_ylim(0, grid[1])
    axP.set_xlabel("x")
    axP.set_ylabel("y")

    # Nets as star-to-centroid
    net_lines = [axP.plot([], [], lw=0.7, alpha=0.45,
                          color="gray")[0] for _ in NETS]

    # Scatter: unlocked (blue) vs locked (orange)
    sc_unlocked = axP.scatter([], [], s=35, marker='s', label="unlocked")
    sc_locked = axP.scatter([], [], s=35, marker='s', label="locked")
    axP.legend(loc="upper right")

    # Obstacle patch (visible in Phase B)
    obs_patches = []
    for (x0, y0, x1, y1) in OBSTACLES:
        r = Rectangle((x0, y0), x1-x0, y1-y0, facecolor="tab:red",
                      alpha=0.12, edgecolor="tab:red")
        r.set_visible(False)
        axP.add_patch(r)
        obs_patches.append(r)

    txtP = axP.text(0.02, 0.98, "", transform=axP.transAxes, va="top")

    axC = fig.add_subplot(gs[0, 1])
    axC.set_title("Cost components")
    axC.set_xlabel("Iteration")
    axC.set_ylabel("Value")
    axC.grid(True, alpha=0.25)
    line_HPWL, = axC.plot([], [], label="HPWL")
    line_DEN, = axC.plot([], [], label="DensityPenalty")
    axC.legend(loc="upper right")

    axT = fig.add_subplot(gs[1, :])
    axT.set_title("Temperature and Total Cost")
    axT.set_xlabel("Iteration")
    axT.grid(True, alpha=0.25)
    line_T,    = axT.plot([], [], label="Temperature (T)")
    line_COST, = axT.plot([], [], label="Total Cost")
    axT.legend(loc="upper right")

    HPWL_vals, DEN_vals = [], []

    def compute_components(pos):
        return hpwl(NETS, pos), density_penalty(pos, grid, bins=(8, 8), target_util=0.8)

    def init():
        sc_unlocked.set_offsets(np.empty((0, 2)))
        sc_locked.set_offsets(np.empty((0, 2)))
        for l in net_lines:
            l.set_data([], [])
        line_HPWL.set_data([], [])
        line_DEN.set_data([], [])
        line_T.set_data([], [])
        line_COST.set_data([], [])
        txtP.set_text("")
        for r in obs_patches:
            r.set_visible(False)
        return [sc_unlocked, sc_locked, *net_lines, line_HPWL, line_DEN, line_T, line_COST, txtP, *obs_patches]

    def update(frame):
        pos = PATH[frame]
        in_phase_b = frame >= PHASE_SPLIT

        # indices for coloring
        unlocked_idx = np.where(
            ~LOCKED)[0] if in_phase_b else np.arange(n_cells)
        locked_idx = np.where(
            LOCKED)[0] if in_phase_b else np.array([], dtype=int)

        sc_unlocked.set_offsets(pos[unlocked_idx])
        sc_locked.set_offsets(pos[locked_idx])

        # nets star-to-centroid
        for ln, pins in zip(net_lines, NETS):
            xs = pos[pins, 0]
            ys = pos[pins, 1]
            cx, cy = xs.mean(), ys.mean()
            xs_plot = np.column_stack(
                [xs, np.full_like(xs, cx)]).ravel(order='F')
            ys_plot = np.column_stack(
                [ys, np.full_like(ys, cy)]).ravel(order='F')
            ln.set_data(xs_plot, ys_plot)

        # components
        hp, dn = compute_components(pos)
        HPWL_vals.append(hp)
        DEN_vals.append(dn)
        its = np.arange(frame+1)
        line_HPWL.set_data(its, HPWL_vals)
        line_DEN.set_data(its, DEN_vals)
        line_T.set_data(its, TEMP[:frame+1])
        line_COST.set_data(its, COST[:frame+1])

        # autoscale occasionally
        if frame % 25 == 0 or frame < 25:
            y1min = min(HPWL_vals)
            y1max = max(HPWL_vals + DEN_vals)
            pad1 = 0.06 * (y1max - y1min + 1e-9)
            axC.set_xlim(0, len(PATH)-1)
            axC.set_ylim(y1min - pad1, y1max + pad1)
            y2min = min(TEMP[:frame+1].min(), COST[:frame+1].min())
            y2max = max(TEMP[:frame+1].max(), COST[:frame+1].max())
            pad2 = 0.06 * (y2max - y2min + 1e-9)
            axT.set_xlim(0, len(PATH)-1)
            axT.set_ylim(y2min - pad2, y2max + pad2)

        for r in obs_patches:
            r.set_visible(in_phase_b)
        phase_lbl = "Phase A (free SA)" if not in_phase_b else "Phase B (reheat + locks + obstacle)"
        txtP.set_text(
            f"{phase_lbl}\niter: {frame}\nT: {TEMP[frame]:.4f}\nTotal: {COST[frame]:.1f}\nHPWL: {hp:.1f}  Den: {dn:.1f}")
        return [sc_unlocked, sc_locked, *net_lines, line_HPWL, line_DEN, line_T, line_COST, txtP, *obs_patches]

    anim = FuncAnimation(fig, update, frames=len(PATH), init_func=init,
                         interval=10, blit=True, repeat=False)

    globals_dict = dict(
        NETS=NETS, GRID=grid, PATH=PATH, COST=COST, TEMP=TEMP,
        LOCKED=LOCKED, OBSTACLES=OBSTACLES, PHASE_SPLIT=PHASE_SPLIT
    )
    return fig, anim, globals_dict

# =========================
# One-liner helper: modify unlocked ones & rerun
# =========================


def rerun_incremental_from_last(globals_dict, lock_idx=(), unlock_idx=(), obstacles=None,
                                iters=900, step=1.2, lam=0.25, reheat=0.18, alpha=0.997, seed=42):
    NETS = globals_dict["NETS"]
    GRID = globals_dict["GRID"]
    PATH = globals_dict["PATH"]
    LOCKED = globals_dict["LOCKED"]
    last_pos = PATH[-1].copy()

    # update locks
    locked = LOCKED.copy()
    if lock_idx:
        locked[list(lock_idx)] = True
    if unlock_idx:
        locked[list(unlock_idx)] = False

    # update obstacles (replace if provided)
    if obstacles is None:
        obstacles = globals_dict["OBSTACLES"]

    PATH2, COST2, TEMP2 = incremental_anneal(
        NETS, last_pos, grid_size=GRID, iters=iters, step=step, lam=lam,
        reheat_factor=reheat, alpha=alpha, locked=locked, obstacles=obstacles, seed=seed
    )

    # swap globals to new phase (only the new run)
    globals_dict["PATH"] = PATH2
    globals_dict["COST"] = COST2
    globals_dict["TEMP"] = TEMP2
    globals_dict["LOCKED"] = locked
    globals_dict["OBSTACLES"] = list(obstacles)
    globals_dict["PHASE_SPLIT"] = 0  # single-phase for the new run

    # Rebuild the animation for the new run
    plt.close('all')
    fig, anim, new_globals = build_animation_from_globals(globals_dict)
    globals_dict.update(new_globals)
    return fig, anim, globals_dict


def build_animation_from_globals(G):
    # builds the same dashboard using arrays in G (for reruns)
    NETS, GRID = G["NETS"], G["GRID"]
    PATH, COST, TEMP = G["PATH"], G["COST"], G["TEMP"]
    LOCKED, OBSTACLES, PHASE_SPLIT = G["LOCKED"], G["OBSTACLES"], G["PHASE_SPLIT"]
    n_cells = PATH.shape[1]

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[
                          2.0, 1.0], hspace=0.30, wspace=0.22)

    axP = fig.add_subplot(gs[0, 0])
    axP.set_title("Placement (cells, nets, obstacle, locked)")
    axP.set_xlim(0, GRID[0])
    axP.set_ylim(0, GRID[1])
    axP.set_xlabel("x")
    axP.set_ylabel("y")

    net_lines = [axP.plot([], [], lw=0.7, alpha=0.45,
                          color="gray")[0] for _ in NETS]
    sc_unlocked = axP.scatter([], [], s=35, marker='s', label="unlocked")
    sc_locked = axP.scatter([], [], s=35, marker='s', label="locked")
    axP.legend(loc="upper right")

    obs_patches = []
    for (x0, y0, x1, y1) in OBSTACLES:
        r = Rectangle((x0, y0), x1-x0, y1-y0, facecolor="tab:red",
                      alpha=0.12, edgecolor="tab:red")
        r.set_visible(False)
        axP.add_patch(r)
        obs_patches.append(r)

    txtP = axP.text(0.02, 0.98, "", transform=axP.transAxes, va="top")

    axC = fig.add_subplot(gs[0, 1])
    axC.set_title("Cost components")
    axC.set_xlabel("Iteration")
    axC.set_ylabel("Value")
    axC.grid(True, alpha=0.25)
    line_HPWL, = axC.plot([], [], label="HPWL")
    line_DEN, = axC.plot([], [], label="DensityPenalty")
    axC.legend(loc="upper right")

    axT = fig.add_subplot(gs[1, :])
    axT.set_title("Temperature and Total Cost")
    axT.set_xlabel("Iteration")
    axT.grid(True, alpha=0.25)
    line_T,    = axT.plot([], [], label="Temperature (T)")
    line_COST, = axT.plot([], [], label="Total Cost")
    axT.legend(loc="upper right")

    HPWL_vals, DEN_vals = [], []

    def compute_components(pos):
        return hpwl(NETS, pos), density_penalty(pos, GRID, bins=(8, 8), target_util=0.8)

    def init():
        sc_unlocked.set_offsets(np.empty((0, 2)))
        sc_locked.set_offsets(np.empty((0, 2)))
        for l in net_lines:
            l.set_data([], [])
        line_HPWL.set_data([], [])
        line_DEN.set_data([], [])
        line_T.set_data([], [])
        line_COST.set_data([], [])
        txtP.set_text("")
        for r in obs_patches:
            r.set_visible(False)
        return [sc_unlocked, sc_locked, *net_lines, line_HPWL, line_DEN, line_T, line_COST, txtP, *obs_patches]

    def update(frame):
        pos = PATH[frame]
        in_phase_b = frame >= PHASE_SPLIT
        n_cells_local = pos.shape[0]

        unlocked_idx = np.where(
            ~LOCKED)[0] if in_phase_b else np.arange(n_cells_local)
        locked_idx = np.where(
            LOCKED)[0] if in_phase_b else np.array([], dtype=int)

        sc_unlocked.set_offsets(pos[unlocked_idx])
        sc_locked.set_offsets(pos[locked_idx])

        for ln, pins in zip(net_lines, NETS):
            xs = pos[pins, 0]
            ys = pos[pins, 1]
            cx, cy = xs.mean(), ys.mean()
            xs_plot = np.column_stack(
                [xs, np.full_like(xs, cx)]).ravel(order='F')
            ys_plot = np.column_stack(
                [ys, np.full_like(ys, cy)]).ravel(order='F')
            ln.set_data(xs_plot, ys_plot)

        hp, dn = compute_components(pos)
        HPWL_vals.append(hp)
        DEN_vals.append(dn)
        its = np.arange(frame+1)
        line_HPWL.set_data(its, HPWL_vals)
        line_DEN.set_data(its, DEN_vals)
        line_T.set_data(its, TEMP[:frame+1])
        line_COST.set_data(its, COST[:frame+1])

        if frame % 25 == 0 or frame < 25:
            y1min = min(HPWL_vals)
            y1max = max(HPWL_vals + DEN_vals)
            pad1 = 0.06 * (y1max - y1min + 1e-9)
            axC.set_xlim(0, len(PATH)-1)
            axC.set_ylim(y1min - pad1, y1max + pad1)
            y2min = min(TEMP[:frame+1].min(), COST[:frame+1].min())
            y2max = max(TEMP[:frame+1].max(), COST[:frame+1].max())
            pad2 = 0.06 * (y2max - y2min + 1e-9)
            axT.set_xlim(0, len(PATH)-1)
            axT.set_ylim(y2min - pad2, y2max + pad2)

        for r in obs_patches:
            r.set_visible(in_phase_b)
        phase_lbl = "Phase A (free SA)" if not in_phase_b else "Phase B (reheat + locks + obstacle)"
        txtP.set_text(
            f"{phase_lbl}\niter: {frame}\nT: {TEMP[frame]:.4f}\nTotal: {COST[frame]:.1f}\nHPWL: {hp:.1f}  Den: {dn:.1f}")
        return [sc_unlocked, sc_locked, *net_lines, line_HPWL, line_DEN, line_T, line_COST, txtP, *obs_patches]

    anim = FuncAnimation(fig, update, frames=len(PATH), init_func=init,
                         interval=10, blit=True, repeat=False)
    new_globals = dict(PATH=PATH, COST=COST, TEMP=TEMP,
                       LOCKED=LOCKED, OBSTACLES=OBSTACLES, PHASE_SPLIT=PHASE_SPLIT)
    return fig, anim, new_globals


# =========================
# Run the default two-phase demo
# =========================
if __name__ == "__main__":
    fig, anim, G = run_demo()
    plt.tight_layout()
    plt.show()

    # --- Example: modify unlocked set and rerun (uncomment to try) ---
    fig2, anim2, G = rerun_incremental_from_last(
        G,
        lock_idx=[1, 5, 9],          # lock a few additional cells
        unlock_idx=[3, 7],          # make these movable again
        obstacles=[(10, 35, 18, 45)],  # new obstacle region
        iters=800, step=1.1, reheat=0.22
    )
    plt.tight_layout()
    plt.show()

    # -------- Optional saving (uncomment one). Requires pillow/ffmpeg. --------
    # anim.save("sa_vlsi_incremental.gif", dpi=130, fps=60)
    # anim.save("sa_vlsi_incremental.mp4", dpi=130, fps=60)
