import random

# Random integer between 1 and 6 (like a dice roll)
print(random.randint(1, 6))     # → 4

# Random even integer between 0 and 10
n = random.randint(0, 5) * 2
print(n)                        # → 8

# Random pixel coordinate
x = random.randint(0, 1920)
y = random.randint(0, 1080)
print(x, y)
