import numpy as np
import matplotlib.pyplot as plt

lam, tmax = 2, 10
arrivals, t = [], 0
while t < tmax:
    t += np.random.exponential(1/lam)
    arrivals.append(t)
plt.vlines(arrivals, 0, 1)
plt.title("Simulated Poisson Process (λ=2)")
plt.xlabel("Time"); plt.ylabel("Events")
plt.savefig("mc_poisson.png", dpi=300)
plt.show()
