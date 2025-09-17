import numpy as np

a = np.array([1.3738729019013636723763], dtype=np.float16)[0]
print("float16:", a)

a = np.array([1.3738729019013636723763], dtype=np.float32)[0]
print("float32:", a)

a = np.array([1.3738729019013636723763], dtype=np.float64)[0]
print("float64:", a)

# float128 is not available on all platforms
#a = np.array([1.3738729019013636723763], dtype=np.float128)[0]
#print("float128:", a)
