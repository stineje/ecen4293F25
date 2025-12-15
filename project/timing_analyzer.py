#!/usr/bin/env python3
"""
timing_analyzer.py

Analyze a Synopsys/Cadence-style timing report for an LZC (or similar) block.

Features:
  - Parse report_timing output.
  - Summarize each timing path.
  - Plot cumulative delay along each path.
  - Build (Cap_load, Fanout, incr_delay) samples by pairing each cell delay
    with the preceding net's load.
  - Run:
        incr_delay ≈ a + b * Cap_load
        incr_delay ≈ a + b * Cap_load + c * Fanout

Intended for a Numerical Methods / Python project using real STA data.
"""

import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Stage:
    point: str
    fanout: Optional[int]
    cap: float
    trans: float
    incr: float
    path: float
    raw_line: str = ""


@dataclass
class TimingPath:
    startpoint: str = ""
    endpoint: str = ""
    group: str = ""
    path_type: str = ""
    operating_conditions: str = ""
    stages: List[Stage] = field(default_factory=list)
    data_arrival_time: Optional[float] = None
    data_required_time: Optional[float] = None
    slack: Optional[float] = None


float_re = re.compile(r"^[+-]?\d+(\.\d+)?$")


def safe_float(s: str) -> Optional[float]:
    s = s.strip()
    return float(s) if float_re.match(s) else None


def parse_timing_report(filename: str) -> List[TimingPath]:
    """
    Parse a timing report into TimingPath objects.

    Assumes:
      - Blocked by 'Startpoint:' ... 'slack'.
      - Table header line contains 'Point', 'Incr', 'Path'.
      - Stage rows may end with 'r'/'f'.
      - Net rows contain '(net)' and have one of:
            (net) fanout cap incr path
            (net) fanout cap      path
            (net)       cap incr path
            (net)       cap      path
      - Cell rows: last numbers are [.., Incr, Path] (and sometimes Trans).
    """
    with open(filename) as f:
        lines = [ln.rstrip("\n") for ln in f]

    paths: List[TimingPath] = []
    cur: Optional[TimingPath] = None
    in_table = False

    for line in lines:
        stripped = line.strip()

        # Start of a new path
        if stripped.startswith("Startpoint:"):
            if cur is not None and cur.stages:
                paths.append(cur)
            cur = TimingPath()
            in_table = False
            cur.startpoint = stripped.split("Startpoint:")[1].strip()
            continue

        if cur is None:
            continue

        # Meta
        if stripped.startswith("Operating Conditions:"):
            cur.operating_conditions = stripped.split("Operating Conditions:")[1].strip()
            continue
        if stripped.startswith("Endpoint:"):
            cur.endpoint = stripped.split("Endpoint:")[1].strip()
            continue
        if stripped.startswith("Path Group:"):
            cur.group = stripped.split("Path Group:")[1].strip()
            continue
        if stripped.startswith("Path Type:"):
            cur.path_type = stripped.split("Path Type:")[1].strip()
            continue

        # Table header
        if stripped.startswith("Point") and "Incr" in stripped and "Path" in stripped:
            in_table = True
            continue

        # Slack line (end of this path block)
        if stripped.startswith("slack"):
            toks = stripped.split()
            cur.slack = safe_float(toks[-1]) if toks else None
            in_table = False
            continue

        # Arrival / required
        low = stripped.lower()
        if low.startswith("data arrival time"):
            toks = stripped.split()
            cur.data_arrival_time = safe_float(toks[-1]) if toks else None
            in_table = False
            continue
        if low.startswith("data required time"):
            toks = stripped.split()
            cur.data_required_time = safe_float(toks[-1]) if toks else None
            continue

        # Skip separators / blanks
        if stripped == "" or set(stripped) == {"-"}:
            continue

        # Stage rows
        if in_table:
            parts = stripped.split()
            if not parts:
                continue

            # Remove trailing r/f marker
            if parts[-1].lower() in ("r", "f"):
                parts = parts[:-1]
            if len(parts) < 2:
                continue

            is_net = "(net)" in parts

            if is_net:
                # ------------- NET ROW (robust) -------------
                i = parts.index("(net)")
                point = " ".join(parts[:i+1])

                # All numeric tokens after "(net)"
                nums = [p for p in parts[i+1:] if float_re.match(p)]

                fanout = None
                cap = 0.0
                incr = 0.0
                pathv = 0.0

                if nums:
                    idx = 0
                    # If the first numeric token is a pure integer, treat as fanout
                    if nums[0].isdigit():
                        fanout = int(nums[0])
                        idx = 1

                    remaining = len(nums) - idx

                    if remaining >= 3:
                        # cap, incr, path
                        cap = float(nums[idx])
                        incr = float(nums[idx + 1])
                        pathv = float(nums[idx + 2])
                    elif remaining == 2:
                        # cap, path
                        cap = float(nums[idx])
                        pathv = float(nums[idx + 1])
                    elif remaining == 1:
                        # only path
                        pathv = float(nums[idx])

                stage = Stage(
                    point=point,
                    fanout=fanout,
                    cap=cap,
                    trans=0.0,
                    incr=incr,
                    path=pathv,
                    raw_line=line,
                )

            else:
                # ------------- CELL / SPECIAL ROW -------------
                # Cell / special row: last numbers are [..., Incr, Path]
                nums = [p for p in parts if float_re.match(p)]
                if not nums:
                    continue

                pathv = float(nums[-1])
                incr = float(nums[-2]) if len(nums) >= 2 else 0.0
                trans = float(nums[-3]) if len(nums) >= 3 else 0.0
                cap = float(nums[-4]) if len(nums) >= 4 else 0.0

                # Point name = tokens before first numeric
                first_num_idx = next(
                    (idx for idx, tok in enumerate(parts) if float_re.match(tok)),
                    len(parts)
                )
                point = " ".join(parts[:first_num_idx])

                stage = Stage(
                    point=point,
                    fanout=None,
                    cap=cap,
                    trans=trans,
                    incr=incr,
                    path=pathv,
                    raw_line=line,
                )

            cur.stages.append(stage)

    # Final path
    if cur is not None and cur.stages:
        paths.append(cur)

    return paths


def debug_stage_stats(paths: List[TimingPath]) -> None:
    total = sum(len(p.stages) for p in paths)
    pos_incr = sum(1 for p in paths for s in p.stages if s.incr > 0)
    nets = sum(1 for p in paths for s in p.stages if "(net)" in s.point)
    print("\n[DEBUG] Parsed stage statistics:")
    print(f"  Total stages     : {total}")
    print(f"  incr > 0         : {pos_incr}")
    print(f"  net stages       : {nets}")


def summarize_path(p: TimingPath) -> None:
    print("=" * 72)
    print(f"Startpoint  : {p.startpoint}")
    print(f"Endpoint    : {p.endpoint}")
    print(f"Path Group  : {p.group}")
    if p.operating_conditions:
        print(f"Conditions  : {p.operating_conditions}")
    if p.path_type:
        print(f"Path Type   : {p.path_type}")

    num_stages = len(p.stages)
    max_incr = max((s.incr for s in p.stages), default=0.0)
    last_path = p.stages[-1].path if p.stages else None

    print(f"Stages      : {num_stages}")
    print(f"Max incr    : {max_incr:.6f} ns")
    if last_path is not None:
        print(f"Last path   : {last_path:.6f} ns")
    if p.data_arrival_time is not None:
        print(f"Arrival     : {p.data_arrival_time:.6f} ns")
    if p.data_required_time is not None:
        print(f"Required    : {p.data_required_time:.6f} ns")
    if p.slack is not None:
        status = "MET" if p.slack >= 0 else "VIOLATED"
        print(f"Slack       : {p.slack:.6f} ns  ({status})")


def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s)


def plot_path(p: TimingPath, show: bool = True, save_prefix: Optional[str] = None) -> None:
    if not p.stages:
        return

    labels = [s.point for s in p.stages]
    delays = [s.path for s in p.stages]

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(delays)), delays, marker="o")
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("Cumulative Delay (ns)")
    title = f"Path: {p.startpoint} → {p.endpoint}"
    if p.slack is not None:
        title += f"  (Slack={p.slack:.6f} ns)"
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_prefix:
        fname = f"{save_prefix}_path_{sanitize_name(p.startpoint)}_{sanitize_name(p.endpoint)}.png"
        plt.savefig(fname, dpi=200)
        print(f"[INFO] Saved plot: {fname}")

    if show:
        plt.show()
    else:
        plt.close()


def collect_cell_load_samples(paths: List[TimingPath]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each cell stage with incr > 0, pair it with the immediately
    preceding net stage (with '(net)' in its point).
    Returns arrays: caps, fanouts, incrs.
    """
    caps = []
    fanouts = []
    incrs = []

    for p in paths:
        prev_net: Optional[Stage] = None
        for s in p.stages:
            if "(net)" in s.point:
                prev_net = s
            else:
                # Treat as cell / pin
                if s.incr > 0 and prev_net is not None:
                    caps.append(prev_net.cap)
                    fanouts.append(float(prev_net.fanout) if prev_net.fanout is not None else 0.0)
                    incrs.append(s.incr)

    return np.array(caps, float), np.array(fanouts, float), np.array(incrs, float)


def perform_least_squares_univariate(paths: List[TimingPath]) -> None:
    caps, _, incrs = collect_cell_load_samples(paths)
    n = len(caps)
    if n < 2:
        print(f"\n[WARN] Not enough data for univariate regression (have {n} samples).")
        return

    A = np.vstack([np.ones_like(caps), caps]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, incrs, rcond=None)
    a, b = coeffs
    pred = A @ coeffs

    ss_res = np.sum((incrs - pred) ** 2)
    ss_tot = np.sum((incrs - np.mean(incrs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print("\n===== Least-Squares Fit (Univariate): cell incr vs net Cap =====")
    print(f"Samples used: {n}")
    print(f"Model: incr_delay = {a:.6f} + {b:.6f} * Cap_load")
    print(f"R^2  : {r2:.4f}")


def perform_least_squares_multivariate(paths: List[TimingPath]) -> None:
    caps, fanouts, incrs = collect_cell_load_samples(paths)
    n = len(caps)
    if n < 3:
        print(f"\n[WARN] Not enough data for multivariate regression (have {n} samples).")
        return

    A = np.vstack([np.ones_like(caps), caps, fanouts]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, incrs, rcond=None)
    a, b, c = coeffs
    pred = A @ coeffs

    ss_res = np.sum((incrs - pred) ** 2)
    ss_tot = np.sum((incrs - np.mean(incrs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print("\n===== Least-Squares Fit (Multivariate): cell incr vs net Cap, Fanout =====")
    print(f"Samples used: {n}")
    print(f"Model: incr_delay = {a:.6f} + {b:.6f} * Cap_load + {c:.6f} * Fanout")
    print(f"R^2  : {r2:.4f}")


def main():
    # FIX: need at least 2 arguments: script name + report file
    if len(sys.argv) < 2:
        print("Usage: python timing_analyzer.py <timing_report.txt>")
        sys.exit(1)

    filename = sys.argv[1]
    paths = parse_timing_report(filename)

    if not paths:
        print("No timing paths parsed. Check report format.")
        sys.exit(1)

    print(f"Parsed {len(paths)} path(s) from {filename}")
    debug_stage_stats(paths)

    # Derive a clean prefix from the report filename
    base_prefix = os.path.splitext(os.path.basename(filename))[0]

    for idx, p in enumerate(paths):
        summarize_path(p)
        plot_path(p, show=(idx == 0), save_prefix=base_prefix)
        # Scatter incr vs Cap with fit, and measured vs predicted
        plot_scatter_cap(paths, save_prefix=base_prefix)
        plot_predicted_vs_measured(paths, save_prefix=base_prefix)

    # Worst slack
    worst = min(
        (p for p in paths if p.slack is not None),
        key=lambda x: x.slack,
        default=None
    )
    if worst is not None:
        print("\nWorst-slack path:")
        print(f"  {worst.startpoint} → {worst.endpoint}, Slack = {worst.slack:.6f} ns")

    # Regression
    perform_least_squares_univariate(paths)
    perform_least_squares_multivariate(paths)
    perform_least_squares_trivariate(paths)

def collect_cell_load_slew_samples(paths: List[TimingPath]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Like collect_cell_load_samples(), but also returns the cell's input transition (slew)
    taken from the *cell* row (Stage.trans). We still pair against the immediately
    preceding net for Cap and Fanout.

    Returns arrays: caps, fanouts, slews, incrs.
    """
    caps = []
    fanouts = []
    slews = []
    incrs = []

    for p in paths:
        prev_net: Optional[Stage] = None
        for s in p.stages:
            if "(net)" in s.point:
                prev_net = s
            else:
                # s is a cell/special row; use its incr and its own trans (input slew)
                if s.incr > 0 and prev_net is not None:
                    caps.append(prev_net.cap)
                    fanouts.append(float(prev_net.fanout) if prev_net.fanout is not None else 0.0)
                    slews.append(s.trans if s.trans is not None else 0.0)
                    incrs.append(s.incr)

    return (np.array(caps, float),
            np.array(fanouts, float),
            np.array(slews, float),
            np.array(incrs, float))    

def perform_least_squares_trivariate(paths: List[TimingPath]) -> None:
    caps, fanouts, slews, incrs = collect_cell_load_slew_samples(paths)
    n = len(caps)
    if n < 4:
        print(f"\n[WARN] Not enough data for Cap+Fanout+Slew regression (have {n} samples).")
        return

    # Design matrix: [1, Cap, Fanout, Slew]
    A = np.vstack([np.ones_like(caps), caps, fanouts, slews]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, incrs, rcond=None)
    a, b, c, d = coeffs
    pred = A @ coeffs

    ss_res = np.sum((incrs - pred) ** 2)
    ss_tot = np.sum((incrs - np.mean(incrs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print("\n===== Least-Squares Fit (Trivariate): incr vs Cap, Fanout, InputSlew =====")
    print(f"Samples used: {n}")
    print(f"Model: incr_delay = {a:.6f} + {b:.6f} * Cap_load + {c:.6f} * Fanout + {d:.6f} * InputSlew")
    print(f"R^2  : {r2:.4f}")

def plot_scatter_cap(paths: List[TimingPath], save_prefix: str) -> None:
    """Scatter: incr_delay vs Cap with univariate fit line."""
    caps, _, incrs = collect_cell_load_samples(paths)
    n = len(caps)
    if n < 2:
        print("[WARN] Not enough samples to draw scatter_cap plot.")
        return

    # Univariate fit: incr = a + b * Cap
    A = np.vstack([np.ones_like(caps), caps]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, incrs, rcond=None)
    a, b = coeffs

    # Line over min..max Cap
    xlin = np.linspace(float(np.min(caps)), float(np.max(caps)), 100)
    ylin = a + b * xlin

    plt.figure(figsize=(7, 5))
    plt.scatter(caps, incrs, s=25, alpha=0.8)
    plt.plot(xlin, ylin, linewidth=2)
    plt.xlabel("Net Capacitance (report units)")
    plt.ylabel("Cell Incremental Delay (ns)")
    plt.title("Incremental Delay vs. Net Capacitance (Univariate Fit)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = f"{save_prefix}_scatter_cap.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[INFO] Saved plot: {out}")


def plot_predicted_vs_measured(paths: List[TimingPath], save_prefix: str) -> None:
    """Measured vs Predicted for best-available linear model (tri > multi > uni)."""
    # Prefer tri-variate -> multi -> uni based on available samples
    caps, fanouts, slews, incrs = collect_cell_load_slew_samples(paths)
    n_tri = len(caps)

    used = None
    if n_tri >= 4:
        # Trivariate: [1, Cap, Fanout, Slew]
        A = np.vstack([np.ones_like(caps), caps, fanouts, slews]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, incrs, rcond=None)
        pred = A @ coeffs
        used = f"Trivariate (Cap, Fanout, Slew) [N={n_tri}]"
    else:
        # Try multivariate
        caps2, fan2, incr2 = collect_cell_load_samples(paths)
        n_multi = len(caps2)
        if n_multi >= 3:
            A = np.vstack([np.ones_like(caps2), caps2, fan2]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, incr2, rcond=None)
            pred = A @ coeffs
            incrs = incr2  # align for plotting
            used = f"Multivariate (Cap, Fanout) [N={n_multi}]"
        else:
            # Fall back to univariate
            caps1, _, incr1 = collect_cell_load_samples(paths)
            n_uni = len(caps1)
            if n_uni < 2:
                print("[WARN] Not enough samples to draw predicted_vs_measured plot.")
                return
            A = np.vstack([np.ones_like(caps1), caps1]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, incr1, rcond=None)
            pred = A @ coeffs
            incrs = incr1
            used = f"Univariate (Cap) [N={n_uni}]"

    # Plot measured vs predicted with y=x reference
    plt.figure(figsize=(7, 5))
    plt.scatter(incrs, pred, s=25, alpha=0.8)
    lims = [min(float(np.min(incrs)), float(np.min(pred))),
            max(float(np.max(incrs)), float(np.max(pred)))]
    plt.plot(lims, lims, linewidth=2)  # y = x
    plt.xlabel("Measured Incremental Delay (ns)")
    plt.ylabel("Predicted Incremental Delay (ns)")
    plt.title(f"Measured vs Predicted Incremental Delay\n{used}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    out = f"{save_prefix}_predicted_vs_measured.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[INFO] Saved plot: {out}")
    
if __name__ == "__main__":
    main()
    
