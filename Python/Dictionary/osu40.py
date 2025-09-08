numbers = [1, 2, 3]
print(*numbers)
print(1, 2, 3)
# Use unapacking operator or iterable
values = list(range(5))
print(values)
# can also unpack this way!
values2 = [*range(5), *"Hello"]
print(values2)
