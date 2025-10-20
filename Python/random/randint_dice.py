import random, matplotlib.pyplot as plt

rolls = [random.randint(1, 6) for _ in range(1000)]
plt.hist(rolls, bins=range(1,8), align='left', color='orange', edgecolor='black')
plt.title("Dice Simulation using randint(1, 6)")
plt.xlabel("Dice Face"); plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("randint_dice.png", dpi=300)
plt.show()
