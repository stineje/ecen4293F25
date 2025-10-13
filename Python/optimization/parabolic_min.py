import numpy as np

def parabolic_min(f, x1, x2, x3, Ea=1e-5, maxit=50, verbose=True):
    """
    Successive parabolic interpolation per Chapra's textbook
    Starts with (x1 < x2 < x3), computes parabola through them,
    and iterates until error <= Ea or maxit reached.
    Returns (xopt, f(xopt), ea, iterations).
    """

    assert x1 < x2 < x3, "Require x1 < x2 < x3 to start."

    f1, f2, f3 = f(x1), f(x2), f(x3)
    ea = np.inf

    # Intelligent printing of dashed line by length of header (match output from Example 7.3)
    if verbose:
        header = f"{'Iter':>4} | {'x1':>10} | {'f(x1)':>10} | {'x2':>10} | {'f(x2)':>10} | {'x3':>10} | {'f(x3)':>10} | {'x4':>10} | {'f(x4)':>10} | {'ea':>10}"
        print(header)
        print("-" * len(header))
        # Iteration 0 (initial guesses only)
        print(f"{0:4d} | {x1:10.6f} | {f1:10.6f} | {x2:10.6f} | {f2:10.6f} | {x3:10.6f} | {f3:10.6f} | {'-':>10} | {'-':>10} | {'-':>10}")

    x2_old = x2
    for i in range(1, maxit + 1):
        # Eq. (7.10)
        num = ((x2 - x1)**2)*(f2 - f3) - ((x2 - x3)**2)*(f2 - f1)
        den = (x2 - x1)*(f2 - f3) - (x2 - x3)*(f2 - f1)

        if den == 0:
            x4 = x2 + 0.25*(x3 - x2) if (f3 < f1) else x2 - 0.25*(x2 - x1)
        else:
            x4 = x2 - 0.5*(num / den)

        # keep x4 in bounds
        if not (x1 < x4 < x3):
            x4 = np.clip(x4, np.nextafter(x1, x3), np.nextafter(x3, x1))
        f4 = f(x4)

        # Chapra’s Strategy Rules (Fig. 7.9)
        if x1 < x4 < x2:
            if f4 < f2:
                x3, f3 = x2, f2
                x2, f2 = x4, f4
            else:
                x1, f1 = x4, f4
        elif x2 < x4 < x3:
            if f4 < f2:
                x1, f1 = x2, f2
                x2, f2 = x4, f4
            else:
                x3, f3 = x4, f4

        ea = abs((x2 - x2_old)/x2) if x2 != 0 else abs(x2 - x2_old)
        x2_old = x2

        if verbose:
            print(f"{i:4d} | {x1:10.6f} | {f1:10.6f} | {x2:10.6f} | {f2:10.6f} | {x3:10.6f} | {f3:10.6f} | {x4:10.6f} | {f4:10.6f} | {ea:10.3e}")

        if ea <= Ea:
            break

    # Intelligent printing of dashed line by length of header
    if verbose:
        print("-" * len(header))

    return x2, f2, ea, i
