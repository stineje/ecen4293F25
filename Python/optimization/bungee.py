from goldmin import goldmin
import matplotlib.pyplot as plt
import numpy as np

g = 9.81 # m/s^2
v0 = 55 # m/s
m = 80 # kg
c = 15 # kg/s
z0 = 100 # m

def f(t):
    return -(z0+m/c*(v0+m*g/c)*(1-np.exp(-t/(m/c)))-m*g/c*t)

tl = 0
tu = 8
tmin, fmin, ea, n = goldmin(f, tl, tu, Ea=1.0e-5)
print('Time at maximum algitute = {0:5.15f} s'.format(tmin))
print('Function value = {0:6.15g}'.format(fmin))
print('Relative error = {0:8.5e}'.format(ea))
print('Number of iterations = {0:5d}'.format(n))

zmax = z0 + m/c*(v0+m*g/c)*(1-np.exp(-tmin/(m/c)))-m*g/c*tmin
print('Maximum altitude = {0:6.15f} m'.format(zmax))
