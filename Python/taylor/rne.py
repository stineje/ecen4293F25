def float_to_binary(x, frac_bits=20):
    """Convert float to binary string with frac_bits fractional bits."""
    # Handle sign externally so we can reuse this for magnitude
    sign = "-" if x < 0 else ""
    x = abs(x)

    int_part = int(x)
    frac = x - int_part

    int_str = bin(int_part)[2:]  # integer part in binary

    frac_str = ""
    for _ in range(frac_bits):
        frac *= 2
        bit = int(frac)
        frac_str += str(bit)
        frac -= bit

    return (sign + f"{int_str}.{frac_str}") if sign else f"{int_str}.{frac_str}"


def round_to_nearest_even_float(x, keep_bits=5, extra_bits=5):
    """
    Round float x to binary with 'keep_bits' fractional bits
    using Python's round() (round-to-nearest-even).

    extra_bits: how many bits beyond keep_bits to generate
                for 'before rounding' binary display.
    """
    binary_str_before = float_to_binary(x, keep_bits + extra_bits)

    # ---- RNE via Python's round() on a 2^keep_bits grid ----
    scale = 1 << keep_bits            # 2**keep_bits
    n = round(x * scale)              # round-half-to-even happens here
    rounded_val = n / scale

    # Rebuild the rounded binary string with exactly keep_bits fractional bits
    sign = "-" if n < 0 else ""
    n_abs = abs(n)
    int_part = n_abs // scale
    frac_part = n_abs % scale

    int_str = bin(int_part)[2:]
    frac_str = format(frac_part, f"0{keep_bits}b") if keep_bits > 0 else ""

    rounded_binary = f"{sign}{int_str}" if keep_bits == 0 else f"{sign}{int_str}.{frac_str}"

    return binary_str_before, rounded_binary, rounded_val


# Output random values and demonstrate RNE
tests = [(3.76, 5), (0.21, 5), (5.3075, 5)]

for val, bits in tests:
    before_bin, after_bin, rounded_val = round_to_nearest_even_float(val, bits, extra_bits=10)
    print(f"Value: {val}")
    print(f"  Binary before rounding: {before_bin}")
    print(f"  Fractional bits kept: {bits}")
    print(f"  Rounded binary: {after_bin}")
    print(f"  Rounded decimal: {rounded_val:.8f}")
    print("-" * 50)
