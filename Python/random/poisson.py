import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda1, tmax = 2, 10  # rate λ=2 events/sec, simulate up to 10 sec

# Generate interarrival and arrival times
interarrivals = np.random.exponential(1/lambda1, 100)
arrivals = np.cumsum(interarrivals)
arrivals = arrivals[arrivals < tmax]

plt.figure(figsize=(8, 2))
plt.vlines(arrivals, ymin=0, ymax=1, color='orange', linewidth=2)
plt.title(f"Simulated Poisson Process (λ={lambda1}, up to t={tmax}s)", fontsize=12)
plt.xlabel("Time (s)")
plt.yticks([])  # remove y-axis since it's just events
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig("poisson_process.png", dpi=300)
plt.show()
