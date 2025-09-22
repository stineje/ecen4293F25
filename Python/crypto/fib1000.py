import sys, math

# Allow very large int → str conversions (0 = unlimited)
sys.set_int_max_str_digits(0)

# Fast-doubling Fibonacci
def fib_fast_doubling(n: int):
    if n == 0:
        return (0, 1)
    a, b = fib_fast_doubling(n >> 1)
    c = a * ((b << 1) - a)           
    d = a*a + b*b                    
    if n & 1:
        return (d, c + d)
    else:
        return (c, d)

n = 1_000_000
Fn, _ = fib_fast_doubling(n)

# Option 1: exact digit count
print(f"F({n}) has {len(str(Fn))} digits.")

# Option 2: if you only need the digit count without full str conversion:
# digits = Fn.bit_length() * math.log10(2)
# print(f"F({n}) has about {int(digits) + 1} digits.")

# Peek at part of the number
s = str(Fn)
print("First 50 digits:", s[:50])
print("Last 50 digits :", s[-50:])
