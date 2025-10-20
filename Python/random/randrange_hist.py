import random, matplotlib.pyplot as plt

data = [random.randrange(1,7) for _ in range(1000)]
plt.hist(data, bins=6, color='orange', edgecolor='black')
plt.title("Random Dice Rolls Using randrange(1,7)")
plt.xlabel("Dice Value"); plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("randrange_hist.png", dpi=300)
plt.show()
