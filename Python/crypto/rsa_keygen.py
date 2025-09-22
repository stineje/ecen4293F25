#!/usr/bin/env python3
# Tiny RSA demo (\approx 2048-bit modulus) using only Python big integers.
# - Keygen with Miller–Rabin probable primes
# - Encrypt/Decrypt (textbook RSA)
# - Sign/Verify (textbook RSA)
# NOTE: For real-world use, always use padding (OAEP / PSS) and a vetted library.

import os
from random import randrange

# ---------- Miller–Rabin probable-prime test ----------
def is_probable_prime(n: int, k: int = 16) -> bool:
    if n < 2:
        return False
    # quick small prime trial division
    small = [2,3,5,7,11,13,17,19,23,29,31]
    for p in small:
        if n % p == 0:
            return n == p
    # write n-1 = d * 2^r with d odd
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    # k random bases
    for _ in range(k):
        a = randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def random_odd(bitlen: int) -> int:
    b = (bitlen + 7) // 8
    x = int.from_bytes(os.urandom(b), "big")
    x |= 1                # force odd
    x |= 1 << (bitlen-1)  # force top bit
    return x

def gen_prime(bitlen: int) -> int:
    while True:
        cand = random_odd(bitlen)
        if is_probable_prime(cand):
            return cand

# ---------- Extended Euclid for modular inverse ----------
def egcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def modinv(a, m):
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("no modular inverse")
    return x % m

# ---------- RSA key generation ----------
def rsa_keygen(bits: int = 2048, e: int = 65537):
    # pick p, q ~ bits/2
    p = gen_prime(bits // 2)
    q = gen_prime(bits // 2)
    while q == p:
        q = gen_prime(bits // 2)
    n = p * q
    phi = (p - 1) * (q - 1)
    d = modinv(e, phi)
    return (n, e, d, p, q)

# ---------- Demo ----------
def main():
    # Keygen
    n, e, d, p, q = rsa_keygen(2048, 65537)

    print("Key sizes:")
    print("  n bits:", n.bit_length())
    print("  p bits:", p.bit_length(), " q bits:", q.bit_length())
    print("  e     :", e)

    # Message as an integer (textbook RSA — NO padding)
    msg = b"hello, big integers!"
    m = int.from_bytes(msg, "big")
    if m >= n:
        raise ValueError("message too large for modulus")

    # Encrypt/Decrypt
    c = pow(m, e, n)
    m2 = pow(c, d, n)
    dec = m2.to_bytes((m2.bit_length() + 7)//8, "big")
    print("Encrypt/Decrypt OK:", dec == msg)

    # Sign/Verify (textbook — NO PSS)
    fake_hash = int.from_bytes(b"demo-hash", "big") % n
    sig = pow(fake_hash, d, n)
    check = pow(sig, e, n)
    print("Signature verifies :", check == fake_hash)

    # Preview modulus without flooding the console
    hx = hex(n)[2:]
    print("n (hex) preview:", hx[:32], "...", hx[-32:])

if __name__ == "__main__":
    main()
    
