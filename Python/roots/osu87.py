import matplotlib.pyplot as plt
import numpy as np


cd = 0.25
g = 9.81
v = 36.0
t = 4.0
mp = np.linspace(50.0, 200.0)
fp = np.sqrt(mp*g/cd)*np.tanh(np.sqrt(g*cd/mp)*t)-v
plt.plot(mp, fp, c='k', lw=0.5)
plt.grid()
plt.xlabel('mass - kg')
plt.ylabel('f(m) - (m/s)')
plt.savefig("fm_vs_mass.png", dpi=300, bbox_inches='tight')
plt.show()
