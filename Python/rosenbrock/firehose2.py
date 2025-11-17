import numpy as np
import matplotlib.pyplot as plt

g = 9.81          # m/s^2
v0 = 15.0         # m/s  (hose exit speed)
h1 = 0.6          # m    (nozzle height)
h2 = 10.0         # m    (roof height)
clearance = 0.0   # m    (set >0 if you want to "just clear" by a margin)
Ntheta = 1000     # angular resolution for the search


def roots_on_roof(theta, v0, h1, h2, g=9.81, clearance=0.0):
    """
    Return the two x-values where the trajectory y(x) intersects the roof height (h2+clearance),
    for a given launch angle theta (radians). If it does not clear, return None.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    if ct <= 1e-12:
        return None  # vertical shot, ignore

    # y(x) = h1 + x*tan(theta) - (g x^2) / (2 v0^2 cos^2 theta)
    # Set y(x) = h2 + clearance and solve ax^2 + bx + c = 0
    a = -g / (2.0 * v0**2 * ct**2)
    b = st / ct
    c = h1 - (h2 + clearance)

    disc = b*b - 4*a*c
    if disc <= 0:
        return None  # no intersection with the roof height

    sqrt_disc = np.sqrt(disc)
    xA = (-b - sqrt_disc) / (2*a)
    xB = (-b + sqrt_disc) / (2*a)
    xs = sorted([xA, xB])

    # Must be in front of the nozzle (x>0) and two distinct hits
    if xs[0] <= 0 or xs[1] <= 0:
        return None
    return xs[0], xs[1]


def coverage_for_theta(theta):
    roots = roots_on_roof(theta, v0, h1, h2, g, clearance)
    if roots is None:
        return None
    x1, x2 = roots
    return (x2 - x1), x1, x2


thetas = np.linspace(np.deg2rad(1), np.deg2rad(89), Ntheta)
best = {"cov": -np.inf, "theta": None, "x1": None, "x2": None}

cov_list, th_kept = [], []
for th in thetas:
    out = coverage_for_theta(th)
    if out is None:
        continue
    cov, x1, x2 = out
    cov_list.append(cov)
    th_kept.append(th)
    if cov > best["cov"]:
        best = {"cov": cov, "theta": th, "x1": x1, "x2": x2}

if best["theta"] is None:
    raise RuntimeError(
        "No angle found that clears the roof. Try adjusting parameters.")

theta_deg = np.rad2deg(best["theta"])
print(f"Optimal angle θ ≈ {theta_deg:.2f}°")
print(f"Front edge distance x1 ≈ {best['x1']:.3f} m")
print(f"Landing distance x2 ≈ {best['x2']:.3f} m")
print(f"Coverage (x2 - x1) ≈ {best['cov']:.3f} m")

# Trajectory for the best angle
th = best["theta"]
ct, st = np.cos(th), np.sin(th)


def y_traj(x):
    return h1 + x*np.tan(th) - (g*x**2)/(2*v0**2*ct**2)


# Plot domain: just beyond landing
xmax = best["x2"] * 1.05
xs = np.linspace(0, xmax, 600)
ys = y_traj(xs)

plt.figure(figsize=(8, 5))
# Roof line
plt.axhline(h2 + clearance, color='k', lw=1.2, label='Roof height')

# Building face at x1 (front edge)
plt.axvline(best["x1"], color='gray', lw=1.2,
            linestyle='--', label='Front edge x1')

# Coverage highlight
plt.fill_between([best["x1"], best["x2"]], h2, h2,
                 color='tab:green', alpha=0.25, label='Coverage')

# Trajectory
plt.plot(xs, ys, label=f"Trajectory at θ={theta_deg:.1f}°")

# Marks
plt.plot([0], [h1], 'ko', label='Nozzle')
plt.plot([best["x1"], best["x2"]], [h2, h2], 'ro', label='Edge/landing')

plt.ylim(0, max(ys.max(), h2) * 1.1)
plt.xlim(0, xmax)
plt.xlabel("Horizontal distance x (m)")
plt.ylabel("Height y (m)")
plt.title("Fire Hose Trajectory and Roof Coverage")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.show()

# Optional: show coverage vs angle curve
if cov_list:
    plt.figure(figsize=(7, 4))
    plt.plot(np.rad2deg(th_kept), cov_list)
    plt.xlabel("Angle θ (degrees)")
    plt.ylabel("Coverage (m)")
    plt.title("Coverage vs. Launch Angle")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
