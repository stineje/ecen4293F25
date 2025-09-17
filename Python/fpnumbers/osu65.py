import numpy as np
import struct

# Example floating-point number
num = np.pi

# Convert to IEEE 754 binary representation (32-bit float)
hex_representation = struct.unpack('<I', struct.pack('<f', num))[0]

# Print as hex
print(f"Hexadecimal representation: {hex_representation:08X}")
