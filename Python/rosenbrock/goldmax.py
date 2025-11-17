import numpy as np


def goldmax(f, xl, xu, Ea=1.e-7, maxit=30):
    """
    Golden-section search to find the MAXIMUM of f(x)
    input:
        f = function to evaluate
        xl = lower bound
        xu = upper bound
        Ea = stopping tolerance (default = 1e-7)
        maxit = maximum iterations (default = 30)
    output:
        xopt = location of the maximum
        f(xopt) = function value at the maximum
        ea = approximate relative error achieved
        iter = number of iterations
    """
    phi = (1 + np.sqrt(5)) / 2
    d = (phi - 1) * (xu - xl)
    x1 = xl + d
    f1 = f(x1)
    x2 = xu - d
    f2 = f(x2)

    for i in range(maxit):
        xint = xu - xl
        if f1 > f2:
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
        if xopt != 0:
            ea = (2 - phi) * abs(xint / xopt)
            if ea <= Ea:
                break

    return xopt, f(xopt), ea, i + 1
