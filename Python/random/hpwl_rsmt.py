import numpy as np
from math import sqrt

# Example nets (x,y pin coordinates)
nets = [
    [(1,1),(4,2),(2,5)],
    [(0,0),(3,3)],
    [(2,2),(2,4),(5,2),(5,4)],
    [(0,0),(5,5)],
    [(1,2),(3,2),(4,6)]
]

def hpwl(net):
    xs, ys = zip(*net)
    return (max(xs)-min(xs)) + (max(ys)-min(ys))

def euclidean(net):
    xs, ys = zip(*net)
    dx, dy = max(xs)-min(xs), max(ys)-min(ys)
    return sqrt(dx**2 + dy**2)

def analyze_net(net):
    e = euclidean(net)
    h = hpwl(net)
    scaled = 0.9 * h
    steiner = 0.75 * h
    hybrid = 0.8*h + 0.2*e
    return {
        'HPWL': h,
        'Scaled': scaled,
        'Hybrid': hybrid,
        'Steiner': steiner,
        'Euclidean': e
    }

# Collect results and compute absolute/percentage errors
print(f"{'Net':<5} {'Euclidean':>10} {'HPWL_err':>10} {'0.9×HPWL_err':>12} {'Hybrid_err':>12} {'Steiner_err':>12}")
print("-"*65)
for i, net in enumerate(nets, 1):
    res = analyze_net(net)
    e = res['Euclidean']
    hpwl_err = abs(res['HPWL'] - e)
    scaled_err = abs(res['Scaled'] - e)
    hybrid_err = abs(res['Hybrid'] - e)
    steiner_err = abs(res['Steiner'] - e)
    print(f"{i:<5} {e:10.3f} {hpwl_err:10.3f} {scaled_err:12.3f} {hybrid_err:12.3f} {steiner_err:12.3f}")
