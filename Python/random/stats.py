import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ----------------------------------------------------------
# Parameters for the normal distribution
# ----------------------------------------------------------
μ, σ = 0, 1     # mean and standard deviation for standard normal

# Define an x-axis grid
x = np.linspace(-4, 4, 400)

# ----------------------------------------------------------
# 1. Probability Density Function (PDF)
# ----------------------------------------------------------
pdf_vals = norm.pdf(x, loc=μ, scale=σ)

# ----------------------------------------------------------
# 2. Cumulative Distribution Function (CDF)
# ----------------------------------------------------------
cdf_vals = norm.cdf(x, loc=μ, scale=σ)

# ----------------------------------------------------------
# 3. Random Sampling (Monte Carlo simulation)
# ----------------------------------------------------------
samples = norm.rvs(loc=μ, scale=σ, size=10000, random_state=0)

# ----------------------------------------------------------
# 4. Theoretical Mean and Variance
# ----------------------------------------------------------
mean, var = norm.stats(moments='mv')
print(f"Theoretical Mean = {mean:.2f}, Variance = {var:.2f}")

# ----------------------------------------------------------
# 5. Plot PDF, CDF, and Histogram
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# --- (a) Left: PDF and histogram of samples ---
axes[0].hist(samples, bins=30, density=True, alpha=0.5, color='orange', label="Sample Histogram")
axes[0].plot(x, pdf_vals, 'k-', linewidth=2, label="Theoretical PDF")
axes[0].set_title("Normal Distribution PDF vs. Sample Histogram")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].grid(alpha=0.3)

# --- (b) Right: CDF plot ---
axes[1].plot(x, cdf_vals, 'b-', linewidth=2)
axes[1].set_title("Cumulative Distribution Function (CDF)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("P(X ≤ x)")
axes[1].grid(alpha=0.3)

plt.tight_layout()

# ----------------------------------------------------------
# 6. Save and Show
# ----------------------------------------------------------
plt.savefig("normal_distribution_demo.png", dpi=300)
plt.show()
