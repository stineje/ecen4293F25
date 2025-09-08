# Create a dictionary of 
point = {"x": 1, "y": 2}

# also use this method instead of L1
# point = dict(x=1, y=2)

# Reassign a value to x
point["x"] = 100
# Add a new entry with key z
point["z"] = 42

# Good programming to avoid issues
if "a" in point:
    print(point["a"])
# Another way to print if you find, if not output 0
print(point.get("a", 0))
# Delete element
del point["x"]

# simple for statement to iterate
for key in point:
    print(key, point[key])
# can also use this way using unpacking for key, value
for key, value in point.items():
    print(key, value)
