import random

# Random number from 0–9
print(random.randrange(10))      # e.g., 7

# Random even number between 0 and 20
print(random.randrange(0, 21, 2))  # e.g., 14

# Random index from a list
items = ['apple', 'orange', 'pear', 'grape']
i = random.randrange(len(items))
print(items[i])  # randomly chosen element
