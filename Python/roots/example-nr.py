# solve_newton_target_0p9.py
import math
from newtraph import newtraph

# F(x) = f(x) - 0.9
def F(x: float) -> float:
    return x - math.exp(-x) * math.sin(x*x) - 0.9

# F'(x) = 1 + e^{-x} sin(x^2) - 2x e^{-x} cos(x^2)
def Fp(x: float) -> float:
    ex = math.exp(-x)
    x2 = x*x
    return 1 + ex*math.sin(x2) - 2*x*ex*math.cos(x2)

# Choose a reasonable initial guess (avoid spots where F'(x) ~ 0)
x0 = 1.0  

x_star, F_at_x, ea, iters = newtraph(F, Fp, x0, Ea=1e-10, maxit=50)

# Report
f_val = x_star - math.exp(-x_star)*math.sin(x_star**2)
print(f"x \\approx {x_star:.12g}")
print(f"f(x) \\approx {f_val:.12g}  (target = 0.9)")
print(f"Residual f(x) - 0.9 \\approx {f_val - 0.9:.3e}")
print(f"Relative error ea ≈ {ea:.3e}, iterations = {iters}")
