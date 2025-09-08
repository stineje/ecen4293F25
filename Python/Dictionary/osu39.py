
from sys import getsizeof
values1 = (x*2 for x in range(1000))
values2 = [x*2 for x in range(1000)]
print("generator size: ", getsizeof(values1))
print("list size: ", getsizeof(values2))

# for x in values1:
#     print(x)
