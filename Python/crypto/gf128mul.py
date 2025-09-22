# GF(2^128) multiplication for AES-GCM (GHASH)
# Polynomial: x^128 + x^7 + x^2 + x + 1
# Implemented via per-step reduction using the reflected constant 0x87 after left shifts.

MASK128 = (1 << 128) - 1
POLY_REFLECT = 0x87  # reduction after a 1-bit left shift within 128 bits

def _xtime(a: int) -> int:
    """Multiply by x in GF(2^128) with reduction (shift-left then conditional XOR)."""
    msb = (a >> 127) & 1
    a = (a << 1) & MASK128
    if msb:
        a ^= POLY_REFLECT
    return a

def gf128_mul(x: int, y: int) -> int:
    """
    Carry-less multiply in GF(2^128) using a 4-bit (nibble) method.
    Processes y four bits at a time with a tiny precompute table for x * {0..15}.
    Returns a 128-bit field element (int in [0, 2^128)).
    """
    # Precompute multiples of current x for nibble 0..15
    T = [0] * 16
    T[0] = 0
    T[1] = x
    for n in range(2, 16):
        if n & 1:
            T[n] = T[n - 1] ^ x
        else:
            T[n] = _xtime(T[n >> 1])

    z = 0
    for _ in range(32):  # 32 nibbles = 128 bits
        nib = y & 0xF
        y >>= 4
        if nib:
            z ^= T[nib]
        # Advance x ← x * (x^4) and refresh the table
        x = _xtime(_xtime(_xtime(_xtime(x))))
        T[0] = 0
        T[1] = x
        for n in range(2, 16):
            if n & 1:
                T[n] = T[n - 1] ^ x
            else:
                T[n] = _xtime(T[n >> 1])
    return z

# ---- Tiny demo / self-check ----
if __name__ == "__main__":
    # Example values similar to AES-GCM usage
    H = int("66e94bd4ef8a2c3b884cfa59ca342b2e", 16)
    X = int("0388dace60b6a392f328c2b971b2fe78", 16)
    Z = gf128_mul(H, X)
    print("H =", f"{H:032x}")
    print("X =", f"{X:032x}")
    print("gf128_mul(H, X) =", f"{Z:032x}")
    
