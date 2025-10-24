# student_A12345678.py  <-- Rename: replace A12345678 with your own ID
# Student Starter File for Lemonade Pricing Competition
# Modify ONLY the my_strategy() function!
# Scoring: Highest Average Profit across 200 fixed-seed trials.

import random

# Pricing bounds (do not change)
MIN_P, MAX_P = 0.50, 2.20


def clamp(x, lo=MIN_P, hi=MAX_P):
    return max(lo, min(hi, x))


def my_strategy(day_index, last_profit, last_price):
    """
    >>> Students modify this function only <<<
    day_index: 0–11
    last_profit: yesterday's profit
    last_price: yesterday's price
    Return: today's price (float)
    """
    # ---------------- REPLACE THIS WITH YOUR STRATEGY ----------------
    # Example: simple 3-tier schedule
    if day_index < 4:
        price = 1.35
    elif day_index < 8:
        price = 1.15
    else:
        price = 0.95
    # Always clamp the price
    return clamp(price)
    # -----------------------------------------------------------------


# ---------- DO NOT MODIFY BELOW THIS LINE ----------
# Deterministic shocks for fairness (fixed sequence)
TEMPS = [84, 78, 90, 95, 68, 84, 72, 90, 78, 95, 90, 72]
RAIN = [0,  1,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0]
NEWS = [0.1, -0.1, 0.2, 0.0, 0.0, 0.1, -0.1, 0.0, 0.2, 0.0, 0.0, 0.0]

COST, FIXED = 0.15, 1.25
DAYS = 12


def demand(price, t, r, n):
    # Linear demand with shocks; Poisson-ish sampling via Gaussian approx
    A, B = 120.0, 35.0
    weather_scale = (t - 60) / 30 + 1
    rain_scale = 0.6 if r else 1.0
    news_scale = 1 + n
    mean = max(0.0, (A - B * price) * weather_scale * rain_scale * news_scale)
    return max(0, int(random.gauss(mean, (mean + 1) ** 0.5)))


def run_season(strategy, seed=0):
    random.seed(seed)
    results = []
    last_profit = 0.0
    last_price = 1.00
    for d in range(DAYS):
        p = clamp(strategy(d, last_profit, last_price))
        D = demand(p, TEMPS[d], RAIN[d], NEWS[d])
        made = int(1.05 * D)
        sold = min(made, D)
        revenue = sold * p
        profit = revenue - made*COST - FIXED
        results.append(profit)
        last_profit = profit
        last_price = p
    return sum(results)


def evaluate(strategy, trials=200):
    total = 0.0
    for k in range(trials):
        total += run_season(strategy, seed=k)
    return total / trials


if __name__ == "__main__":
    avg = evaluate(my_strategy, trials=200)
    print(f"Avg Profit = ${avg:.2f}")
