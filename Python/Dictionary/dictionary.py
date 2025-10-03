# string is a key and integer for the value in the dictionary
point = {"x": 1, "y": 2}
point = dict(x=1, y=2)  # another way to create a dictionary
point["x"] = 42  # update value for key "x"
point["z"] = 1890  # add a new key-value pair
print(point)  # Output: {'x': 42, 'y': 2, 'z': 1890} 3 key-value pairs
if "a" in point:  # check if key exists otherwise might throw an error
    print(point["a"])  # print value if key exists
# get value for key "a" or return 0 (default) if key doesn't exist
print(point.get("a", 0))
del point["x"]  # delete a key-value pair
print(point)  # Output: {'y': 2, 'z': 1890} 2 key-value pairs

for key in point:  # iterable over keys in the dictionary
    print(key, point[key])  # iterate over keys and print key-value pairs

# another method to iterate over key-value pairs
for key, value in point.items():  # iterable over key-value pairs
    print(key, value)  # prints tuples of (key, value)
