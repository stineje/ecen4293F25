import numpy as np
import matplotlib.pyplot as plt

def huber_loss(residual, c):
    """
    Compute the Huber loss for a given residual and threshold c.
    """
    abs_residual = np.abs(residual)
    return np.where(abs_residual <= c, 0.5 * residual ** 2, c * (abs_residual - 0.5 * c))

# Define the range of residuals
residuals = np.linspace(-5, 5, 1000)

# Define different values of c to plot
c_values = np.linspace(0.5, 5, 10)  # 10 values of c between 0.5 and 5

# Plot Huber loss for different values of c
plt.figure(figsize=(10, 6))

for c in c_values:
    loss = huber_loss(residuals, c=c)
    plt.plot(residuals, loss, label=f'c = {c:.2f}')

# Add labels, title, and legend
plt.title("Huber's Loss Function for Different Values of c", fontsize=15)
plt.xlabel("Residual", fontsize=12)
plt.ylabel("Huber Loss", fontsize=12)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True)
plt.legend(title="Threshold c")

# Show the plot
plt.show()
