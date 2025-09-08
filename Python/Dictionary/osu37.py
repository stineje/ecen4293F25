# list comprehension
values1 = {x*2 for x in range(5)}
# dictionaries use {} but have key value pairs
values2 = {x: x*2 for x in range(5)}
print("Sets: ", values1)
print("Dictionaries: ", values2)
