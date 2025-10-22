import numpy as np
from math import sqrt

# Example nets: each net is a list of (x, y) pin coordinates
nets = [
    [(1, 1), (4, 2), (2, 5)],           # net1
    [(0, 0), (3, 3)],                   # net2
    [(2, 2), (2, 4), (5, 2), (5, 4)]    # net3
]

# Half-Perimeter Wirelength (bounding box width + height)
def hpwl(net):
    xs, ys = zip(*net)
    return (max(xs) - min(xs)) + (max(ys) - min(ys))

# Euclidean Wirelength (hypotenuse of bounding box)
def euclidean_wirelength(net):
    xs, ys = zip(*net)
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    return sqrt(dx**2 + dy**2)

# Compare HPWL vs Euclidean
print(f"{'Net':<5} {'HPWL':>8} {'Euclidean':>12} {'Ratio (HPWL/Euc)':>18}")
print("-" * 45)
for i, n in enumerate(nets, 1):
    h = hpwl(n)
    e = euclidean_wirelength(n)
    ratio = h / e if e != 0 else float('inf')
    print(f"{i:<5} {h:8.3f} {e:12.3f} {ratio:18.3f}")
