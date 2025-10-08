import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return x - np.exp(-x) * np.sin(x**2)

# Define a range of x values
x = np.linspace(-3, 3, 600)
y = f(x)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, color='darkblue', label=r"$f(x) = x - e^{-x}\sin(x^2)$")
plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # horizontal zero line
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Plot of $f(x) = x - e^{-x} \\sin(x^2)$")
plt.legend()
plt.grid(True)
plt.savefig("example.png", dpi=300, bbox_inches='tight')
plt.show()
