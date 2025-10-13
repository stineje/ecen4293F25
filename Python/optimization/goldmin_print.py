import numpy as np

def goldmin_print(f, xl, xu, Ea=1.e-7, maxit=30):
    """
    Golden-section search to find the minimum of f(x).
    Prints each iteration with xl, xu, x1, x2, f(x1), f(x2), d, and ea.
    """
    phi = (1 + np.sqrt(5)) / 2

    # Initial setup
    d = (phi - 1) * (xu - xl)
    x1 = xl + d
    x2 = xu - d
    f1 = f(x1)
    f2 = f(x2)
    ea = np.inf

    # Header line
    header = f"{'Iter':>4} | {'xl':>10} | {'xu':>10} | {'x1':>10} | {'x2':>10} | {'f(x1)':>10} | {'f(x2)':>10} | {'d':>10} | {'ea':>10}"
    print(header)
    print("-" * len(header))

    # Print initial condition
    print(f"{0:4d} | {xl:10.6f} | {xu:10.6f} | {x1:10.6f} | {x2:10.6f} | {f1:10.6f} | {f2:10.6f} | {d:10.6f} | {'-':>10}")

    # Iterations
    for i in range(maxit):
        xint = xu - xl

        if f1 < f2:
            xopt = x1
            xl = x2
            x2 = x1
            f2 = f1
            x1 = xl + (phi - 1) * (xu - xl)
            f1 = f(x1)
        else:
            xopt = x2
            xu = x1
            x1 = x2
            f1 = f2
            x2 = xu - (phi - 1) * (xu - xl)
            f2 = f(x2)

        d = (phi - 1) * (xu - xl)
        ea = (2 - phi) * abs(xint / xopt) if xopt != 0 else np.nan

        print(f"{i+1:4d} | {xl:10.6f} | {xu:10.6f} | {x1:10.6f} | {x2:10.6f} | {f1:10.6f} | {f2:10.6f} | {d:10.6f} | {ea:10.3e}")

        if ea <= Ea:
            break

    print("-" * len(header))
    print(f"Optimal x = {xopt:.8f},  f(xopt) = {f(xopt):.8f},  ea = {ea:.3e},  iterations = {i+1}")

    return xopt, f(xopt), ea, i + 1
