# econ_lemonade_A12345678.py  <-- Rename: replace A12345678 with your own ID
# Student Starter File for Lemonade Pricing Competition
# Modify ONLY the my_strategy() function!
# Scoring: Highest Average Profit across 200 fixed-seed trials.

import random
import matplotlib.pyplot as plt
import numpy as np

# Pricing bounds (do not change)
MIN_P, MAX_P = 0.50, 2.20


def clamp(x, lo=MIN_P, hi=MAX_P):
    return max(lo, min(hi, x))


def my_strategy(day_index, last_profit, last_price):
    price = last_price

    # First day: start mid-range
    if day_index == 0:
        return 1.20

    # If profit fell → lower price by 10¢
    if last_profit < 0:
        price -= 0.10
    # If profit was good → raise price by 5¢
    elif last_profit > 1.00:
        price += 0.05

    # Clamp to allowed range
    return max(0.50, min(2.20, price))


def visualize_strategy(strategy):
    # Run ONE season and track price + profit
    random.seed(99)
    prices = []
    profits = []
    last_profit = 0.0
    last_price = 1.00
    for d in range(DAYS):
        p = clamp(strategy(d, last_profit, last_price))
        D = demand(p, TEMPS[d], RAIN[d], NEWS[d])
        made = int(1.05 * D)
        sold = min(made, D)
        revenue = sold * p
        profit = revenue - made*COST - FIXED

        prices.append(p)
        profits.append(profit)

        last_profit = profit
        last_price = p

    # Dual plot: price + profit
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Price ($)", color="tab:orange")
    ax1.plot(range(1, DAYS+1), prices, marker="o")
    ax1.tick_params(axis="y", labelcolor="tab:orange")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Profit ($)", color="tab:blue")
    ax2.plot(range(1, DAYS+1), profits, marker="x", linestyle="--")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Strategy Performance (1 season)")
    fig.tight_layout()
    plt.show()


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
    mean = max(0.0, (A - B * price)*weather_scale*rain_scale*news_scale)
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

    visualize_strategy(my_strategy)
