import numpy as np

array1_in = np.array([1., 2., 3.])
array1_cm = array1_in*2.54
# using list comprehension
array2_in = [1, 2, 3]
array2_cm = [x * 2.54 for x in array2_in]
print(array2_in, array2_cm)