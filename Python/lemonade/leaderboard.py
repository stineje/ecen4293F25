# leaderboard.py
# Leaderboard Evaluator (Option A: Highest Average Profit)
# Usage:
#   python econ_lemonade_teacher.py --dir submissions --trials 200
# Expect student files named like: student_A12345678.py

import argparse
import os
import glob
import importlib.util
import statistics

# Fixed shocks for fairness (must match student files)
TEMPS = [84, 78, 90, 95, 68, 84, 72, 90, 78, 95, 90, 72]
RAIN = [0,  1,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0]
NEWS = [0.1, -0.1, 0.2, 0.0, 0.0, 0.1, -0.1, 0.0, 0.2, 0.0, 0.0, 0.0]

MIN_P, MAX_P = 0.50, 2.20


def clamp(x, lo=MIN_P, hi=MAX_P):
    return max(lo, min(hi, x))


def demand(random, price, t, r, n):
    A, B = 120.0, 35.0
    weather_scale = (t - 60) / 30 + 1
    rain_scale = 0.6 if r else 1.0
    news_scale = 1 + n
    mean = max(0.0, (A - B * price)*weather_scale*rain_scale*news_scale)
    return max(0, int(random.gauss(mean, (mean + 1) ** 0.5)))


def run_season(random, strategy, seed=0, days=12, cost=0.15, fixed=1.25):
    random.seed(seed)
    last_profit = 0.0
    last_price = 1.00
    total = 0.0
    for d in range(days):
        p = clamp(float(strategy(d, last_profit, last_price)))
        D = demand(random, p, TEMPS[d], RAIN[d], NEWS[d])
        made = int(1.05 * D)
        sold = min(made, D)
        revenue = sold * p
        profit = revenue - made*cost - fixed
        total += profit
        last_profit = profit
        last_price = p
    return total


def evaluate(random, strategy, trials=200):
    scores = []
    for k in range(trials):
        scores.append(run_season(random, strategy, seed=k))
    avg = sum(scores)/len(scores)
    sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return avg, sd


def load_strategy(path):
    module_name = os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "my_strategy"):
        raise AttributeError(
            f"{path} has no my_strategy(day,last_profit,last_price) function")
    return mod.my_strategy, module_name


def main():
    import random as pyrandom
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".",
                        help="Directory containing student files")
    parser.add_argument("--trials", type=int, default=200,
                        help="Monte Carlo trials per submission")
    parser.add_argument("--pattern", default="student_",
                        help="Filename prefix to include")
    args = parser.parse_args()

    files = [os.path.join(args.dir, f) for f in os.listdir(args.dir)
             if f.startswith(args.pattern) and f.endswith(".py") and "TEMPLATE" not in f]

    if not files:
        print("No student files found. Expect names like student_A12345678.py")
        return

    results = []
    for fp in sorted(files):
        try:
            strat, name = load_strategy(fp)
            avg, sd = evaluate(pyrandom, strat, trials=args.trials)
            results.append((avg, sd, name, fp))
            print(f"Evaluated {name:30s}  Avg=${avg:8.2f}  SD=${sd:7.2f}")
        except Exception as e:
            print(f"[ERROR] {fp}: {e}")

    # Sort by average profit (descending). Ties: by lower SD first
    results.sort(key=lambda t: (-t[0], t[1], t[2]))

    # Print leaderboard
    print("\n Leaderboard — Highest Avg Profit Wins ")
    print("-------------------------------------------------------------")
    print(f"{'Rank':<5}{'Student File':<32}{'Avg Profit':>12}{'SD':>10}")
    print("-------------------------------------------------------------")
    for i, (avg, sd, name, fp) in enumerate(results, 1):
        print(f"{i:<5}{name:<32}${avg:>11.2f}{sd:>10.2f}")

    # Save CSV
    out_csv = os.path.join(args.dir, "leaderboard.csv")
    with open(out_csv, "w") as f:
        f.write("rank,filename,avg_profit,sd\n")
        for i, (avg, sd, name, fp) in enumerate(results, 1):
            f.write(f"{i},{name},{avg:.2f},{sd:.2f}\n")
    print(f"\nSaved leaderboard to: {out_csv}")


if __name__ == "__main__":
    main()
