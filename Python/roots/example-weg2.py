import math
from wegstein import wegstein

# f(x) for checking
def f(x):
    return x - math.exp(-x)*math.sin(x*x)

target = 0.9

# Fixed-point map for: 0.9 = x - exp(-x)*sin(x^2)  =>  x = 0.9 + exp(-x)*sin(x^2)
def g(x):
    return target + math.exp(-x) * math.sin(x*x)

# Start away from 0 to avoid your wegstein's ea division-by-zero quirk
x0, x1 = 0.9, 1.2

x_star, ea, iters = wegstein(g, x0, x1, Ea=1e-10, maxit=50)

print(f"x \\approx {x_star:.12g}")
print(f"Check: f(x) = {f(x_star):.12g}  (target={target})")
print(f"Residual f(x)-target = {f(x_star) - target:.3e}")
print(f"ea ≈ {ea:.3e}, iterations = {iters}")
