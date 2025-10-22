import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# --- Parameters ---
λ, t = 3, 1        # rate λ=3 events/sec, over interval of 1 sec
N = 100_000        # number of Monte Carlo trials

# --- Monte Carlo Simulation ---
counts = np.random.poisson(λ*t, N)
p_est = np.mean(counts >= 4)

print(f"Monte Carlo Estimate: P(N≥4) ≈ {p_est:.4f}")

# --- Theoretical Distribution ---
k = np.arange(0, np.max(counts)+1)
pmf = poisson.pmf(k, λ*t)

# --- Plot Histogram vs. Theory ---
plt.figure(figsize=(7,4))
plt.hist(counts, bins=np.arange(-0.5, np.max(counts)+1.5, 1),
         density=True, alpha=0.6, color='orange', label='Simulated Counts')
plt.plot(k, pmf, 'ko-', label='Theoretical Poisson PMF')

plt.title(f"Monte Carlo Poisson Simulation (λ={λ}, t={t})", fontsize=12)
plt.xlabel("Event Count N(t)")
plt.ylabel("Probability")
plt.legend()
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

# --- Save Plot to File ---
plt.savefig("monte_carlo_poisson.png", dpi=300)
plt.show()
