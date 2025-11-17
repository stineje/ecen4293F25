import numpy as np
from goldmax import goldmax

g = 9.81
v0 = 15.0
h1 = 0.6
h2 = 10.0
H = h2 - h1


def coverage(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    if c <= 1e-12:
        return -np.inf
    a = -g / (2.0 * v0**2 * c**2)
    b = s / c
    c0 = h1 - h2
    D = b*b - 4*a*c0
    if D <= 0:
        return -np.inf
    # coverage = difference of roots
    return np.sqrt(D) / abs(a)


# search angles that are feasible (about 65–85 degrees here)
th_l, th_u = np.radians(65), np.radians(85)
th_opt, cov_opt, ea, n = goldmax(coverage, th_l, th_u, Ea=1e-6)

# compute x1, x2 at the optimum
c = np.cos(th_opt)
s = np.sin(th_opt)
a = -g / (2.0 * v0**2 * c**2)
b = s/c
c0 = h1 - h2
D = b*b - 4*a*c0
r1 = (-b - np.sqrt(D))/(2*a)
r2 = (-b + np.sqrt(D))/(2*a)

print(f"theta* = {np.degrees(th_opt):.3f} deg")
print(f"x1 = {r1:.3f} m, x2 = {r2:.3f} m, coverage = {cov_opt:.3f} m")
