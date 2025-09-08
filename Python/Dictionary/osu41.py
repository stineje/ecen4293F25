first = [1, 2]
second = [3]
values = [*first, "Go Pokes!", *second, *"Hello"]
print(values)
third = {"x": 1}
fourth = {"x": 10, "y": 2}
combined = {**third, **fourth, "z": 42}
print(combined)
