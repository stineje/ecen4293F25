numbers = [1, 1, 1, 2, 3, 3, 4, 3, 3]
first = set(numbers)
second = {1, 5}
# union
print(first | second)
# intersection
print(first & second)
# difference
print(first - second)
# geometric difference
print(first ^ second)
