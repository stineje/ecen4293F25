import numpy as np
import matplotlib.pyplot as plt

# Define the adjacency matrix (transition probabilities)
adjacency_matrix = np.array([
    [0, 0.025, 0.45, 0.025, 0.025, 0.025],  # Google
    [0.025, 0, 0.167, 0.025, 0.025, 0.025],  # Facebook
    [0.45, 0.167, 0, 0.167, 0.167, 0.167],  # YouTube
    [0.025, 0.025, 0.167, 0, 0.025, 0.025],  # Twitter
    [0.025, 0.025, 0.167, 0.167, 0, 0.308],  # Wikipedia
    [0.025, 0.025, 0.167, 0.025, 0.308, 0]  # Amazon
])

# Number of time steps
n_steps = 15

# Initial state (distribution of users across websites)
initial_state = np.array([1, 0, 0, 0, 0, 0])  # All users start at Google

# To store probabilities at each time step
probabilities = [initial_state]

# Compute the probabilities at each time step
for _ in range(n_steps):
    next_state = np.dot(probabilities[-1], adjacency_matrix)
    probabilities.append(next_state)

# Convert probabilities list to numpy array for easier manipulation
probabilities = np.array(probabilities)

# Plotting the results
plt.figure(figsize=(10, 6))
labels = ['Google', 'Facebook', 'YouTube', 'Twitter', 'Wikipedia', 'Amazon']

for i in range(len(labels)):
    plt.plot(probabilities[:, i], label=labels[i])

plt.title('Probabilities of User at Each Website Over Time')
plt.xlabel('Time Step')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.savefig("pagerank_xc.png", dpi=300, bbox_inches='tight')
plt.show()
