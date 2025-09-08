list1 = [1, 2, 3]
list2 = [4, 5, 6]
# combining multiple lists
z1 = [(1, 4), (2, 5), (3, 6)]
# Zip (use list because its an object that it returns4)
z2 = list(zip("OSU", list1, list2))
print(z1)
print(z2)
