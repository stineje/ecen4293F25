import numpy as np, matplotlib.pyplot as plt

rng = np.random.default_rng(0)
fig, axs = plt.subplots(1,3,figsize=(7,2.3))
axs[0].hist(rng.normal(0,1,1000),30,color='orange'); axs[0].set_title("Normal")
axs[1].hist(rng.uniform(-1,1,1000),30,color='orange'); axs[1].set_title("Uniform")
axs[2].hist(rng.exponential(1,1000),30,color='orange'); axs[2].set_title("Exponential")
plt.tight_layout()
plt.savefig("random_dists.png",dpi=300)
