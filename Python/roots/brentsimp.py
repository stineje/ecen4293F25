import numpy as np


def brentsimp(f, xl, xu):
    eps = np.finfo(float).eps
    a = xl
    b = xu
    fa = f(a)
    fb = f(b)
    c = a
    fc = fa
    d = b - c
    e = d
    while True:
        if fb == 0:
            break
        if np.sign(fa) == np.sign(fb):  # rearrange points as req'd
            a = c
            fa = fc
            d = b - c
            e = d
        if abs(fa) < abs(fb):
            c = b
            b = a
            a = c
            fc = fb
            fb = fa
            fa = fc
        m = (a-b)/2  # termination test and possible exit
        tol = 2 * eps * max(abs(b), 1)
        if abs(m) < tol or fb == 0:
            break
        # choose open methods or bisection
        if abs(e) >= tol and abs(fc) > abs(fb):
            s = fb/fc
            if a == c:
                # secant method here
                p = 2*m*s
                q = 1 - s
            else:
                # inverse quadratic interpolation here
                q = fc/fa
                r = fb/fa
                p = s * (2*m*q*(q-r)-(b-c)*(r-1))
                q = (q-1)*(r-1)*(s-1)
            if p > 0:
                q = -q
            else:
                p = -p
            if 2*p < 3*m*q - abs(tol*q) and p < abs(0.5*e*q):
                e = d
                d = p/q
            else:
                d = m
                e = m
        else:
            # bisection here
            d = m
            e = m
        c = b
        fc = fb
        if abs(d) > tol:
            b = b + d
        else:
            b = b - np.sign(b-a)*tol
        fb = f(b)
    return b


