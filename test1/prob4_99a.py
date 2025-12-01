# Problem 4.99 - Richardson Extrapolation for f'(2) and f''(2)
# James E. Stine
# Oklahoma State University

def f(x):
    return 25*x**3 - 6*x**2 + 7*x - 88


def fprime_true(x):
    # derivative of f: 75x^2 - 12x + 7
    return 75*x**2 - 12*x + 7


def fsecond_true(x):
    # second derivative of f: 150x - 12
    return 150*x - 12


def D_centered(f, x0, h):
    """
    Approximate f'(x0) using the centered difference formula:
        D(h) = (f(x0+h) - f(x0-h)) / (2h)
    This uses the function values on both sides of x0.
    """
    return (f(x0 + h) - f(x0 - h)) / (2*h)


def D2_threepoint(f, x0, h):
    """
    Approximate f''(x0) using the three-point formula:
        f''(x0) ≈ (f(x0+h) - 2f(x0) + f(x0-h)) / h^2
    This is called 'three-point' because it uses exactly
    three function values: f(x0-h), f(x0), and f(x0+h).
    """
    return (f(x0 + h) - 2*f(x0) + f(x0 - h)) / (h*h)


def richardson(Dh, Dh2, p):
    """
    Combine two approximations with step sizes h and h/2
    to remove the leading error term:
        D_rich = (2^p * D(h/2) - D(h)) / (2^p - 1)
    """
    return (2**p * Dh2 - Dh) / (2**p - 1)


def abs_err(approx, truth):
    return abs(approx - truth)


def pct_true_rel_err(approx, truth):
    return 0.0 if truth == 0 else 100.0 * abs(approx - truth) / abs(truth)


# --- Evaluate for x0 = 2, h = 0.4 and 0.2 ---
x0 = 2.0
h_big, h_small = 0.4, 0.2

# True values
fp_true = fprime_true(x0)
fpp_true = fsecond_true(x0)

# First derivative
D_h = D_centered(f, x0, h_big)
D_h2 = D_centered(f, x0, h_small)
D_rich = richardson(D_h, D_h2, p=2)

# Second derivative (three-point formula)
S_h = D2_threepoint(f, x0, h_big)
S_h2 = D2_threepoint(f, x0, h_small)
S_rich = richardson(S_h, S_h2, p=2)


def line(name, val, truth):
    """ report results """
    print(f"{name:16s} = {val:12.6f} | abs err = {abs_err(val, truth):9.6f} | "
          f"% true rel err = {pct_true_rel_err(val, truth):9.6f}")


print("f'(2) results (centered difference & Richardson, p=2):")
line("True f'(2)", fp_true, fp_true)
line("D(h=0.4)",  D_h,    fp_true)
line("D(h=0.2)",  D_h2,   fp_true)
line("D_rich",    D_rich, fp_true)

print("\nf''(2) results (three-point formula & Richardson, p=2):")
line("True f''(2)", fpp_true, fpp_true)
line("S(h=0.4)",   S_h,      fpp_true)
line("S(h=0.2)",   S_h2,     fpp_true)
line("S_rich",     S_rich,   fpp_true)
