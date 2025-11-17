import numpy as np


def goldmin_error(f, xl, xu, Ea=1e-7):
    """
    Golden-section search to find the MINIMUM of f(x).
    Instead of using Eq. (7.9) or a maximum iteration stop,
    it computes exactly how many iterations are required to
    achieve a desired absolute x-tolerance (Ea).

    Prints each iteration showing xl, xu, x1, x2, f(x1), f(x2), d, and ea.
    """

    phi = (1 + np.sqrt(5)) / 2
    r = phi - 1  # = 1/phi ≈ 0.618

    # --- Determine required iterations to reach tolerance Ea ---
    L0 = xu - xl
    if L0 <= 2.0 * Ea:
        n_req = 0
    else:
        n_req = int(np.ceil(np.log(2.0 * Ea / L0) / np.log(r)))
        n_req = max(n_req, 0)

    d = r * (xu - xl)
    x1 = xl + d
    x2 = xu - d
    f1 = f(x1)
    f2 = f(x2)

    # --- Print header ---
    header = f"{'Iter':>4} | {'xl':>10} | {'xu':>10} | {'x1':>10} | {'x2':>10} | {'f(x1)':>10} | {'f(x2)':>10} | {'d':>10} | {'ea':>10}"
    print(header)
    print("-" * len(header))

    # Initial print
    print(f"{0:4d} | {xl:10.6f} | {xu:10.6f} | {x1:10.6f} | {x2:10.6f} | {f1:10.6f} | {f2:10.6f} | {d:10.6f} | {'-':>10}")

    # --- Perform exactly n_req iterations ---
    for i in range(n_req):
        xint = xu - xl

        if f1 < f2:
            xl = x2
            x2 = x1
            f2 = f1
            x1 = xl + r * (xu - xl)
            f1 = f(x1)
        else:
            xu = x1
            x1 = x2
            f1 = f2
            x2 = xu - r * (xu - xl)
            f2 = f(x2)

        d = r * (xu - xl)
        ea = 0.5 * (xu - xl)

        print(f"{i+1:4d} | {xl:10.6f} | {xu:10.6f} | {x1:10.6f} | {x2:10.6f} | "
              f"{f1:10.6f} | {f2:10.6f} | {d:10.6f} | {ea:10.3e}")

    # --- Final result ---
    xopt = 0.5 * (xl + xu)
    fopt = f(xopt)
    ea_final = 0.5 * (xu - xl)

    print("-" * len(header))
    print(
        f"Optimal x = {xopt:.8f},  f(xopt) = {fopt:.8f},  ea = {ea_final:.3e},  iterations = {n_req}")

    return xopt, fopt, ea_final, n_req
