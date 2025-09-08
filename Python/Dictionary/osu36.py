values1 = []
for x in range(5):
    values1.append(x*2)

# Easier using list comprehension
# [expresion for item in items]
values2 = values = [x*2 for x in range(5)]
print(values1)
# gives the same value and code is shorter
print(values2)
