import numpy as np
import struct
import math

# Example double-precision floating-point number
num = np.pi

# Convert to IEEE 754 binary representation (64-bit double)
hex_representation = struct.unpack('<Q', struct.pack('<d', num))[0]

# Print as hexadecimal
print(f"Hexadecimal representation: {hex_representation:016X}")


print(format(1/10, '.299g'))
