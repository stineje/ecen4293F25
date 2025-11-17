from goldmax import goldmax
import matplotlib.pyplot as plt
import numpy as np

g = 9.81  # m/s^2
v0 = 55   # m/s
m = 80    # kg
c = 15    # kg/s
z0 = 100  # m


def f(t):
    return -(z0 + m/c*(v0 + m*g/c)*(1 - np.exp(-t/(m/c))) - m*g/c*t)


def f_neg(t): return -f(t)


tl = 0
tu = 8

tmax, fmax, ea, n = goldmax(f_neg, tl, tu, Ea=1.0e-5)

print('Time at maximum altitude = {0:5.15f} s'.format(tmax))
print('Function value = {0:6.15g}'.format(fmax))
print('Relative error = {0:8.5e}'.format(ea))
print('Number of iterations = {0:5d}'.format(n))

# Compute maximum altitude
zmax = z0 + m/c*(v0 + m*g/c)*(1 - np.exp(-tmax/(m/c))) - m*g/c*tmax
print('Maximum altitude = {0:6.15f} m'.format(zmax))

# --- Plotting section ---
t = np.linspace(tl, tu, 400)
zt = z0 + m/c*(v0 + m*g/c)*(1 - np.exp(-t/(m/c))) - m*g/c*t

plt.figure(figsize=(7, 5))
plt.plot(t, zt, 'b-', label='z(t)')
plt.plot(tmax, zmax, 'ro', label='Maximum altitude')
plt.title('Maximum Altitude with Linear Drag (Golden-Section Search using goldmax)')
plt.xlabel('t (s)')
plt.ylabel('z(t) (m)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
