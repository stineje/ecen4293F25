import math
from wegstein import wegstein

def f(x):
    return x - math.exp(-x)*math.sin(x*x)

x0, x1 = 0.3, 1.0  
x_star, ea, iters = wegstein(f, x0, x1, Ea=1e-10, maxit=50)

print(f"x \\approx {x_star:.12g}, f(x) \\approx {f(x_star):.3e}, ea \\approx {ea:.3e}, iterations = {iters}")
