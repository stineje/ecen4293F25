import numpy as np
import struct

x = np.pi
print(x)
print(x.as_integer_ratio())
x_is_what = (x == (884279719003555 / 281474976710656))
print(x_is_what)
