import numpy as np


def newtraph(f, fp, x0, Ea=1.e-7, maxit=30):
    """
    This function solves f(x)=0 using the Newton-Raphson method.
    The method is repeated until either the relative error
    falls below Ea (default 1.e-7) or reaches maxit (default 30).
    Input:
        f = name of the function for f(x)
        fp = name of the function for f'(x)
        x0 = initial guess for x
        Ea = relative error threshold
        maxit = maximum number of iterations
    Output:
        x1 = solution estimate
        f(x1) = equation error at solution estimate
        ea = relative error
        i+1 = number of iterations
    """
    for i in range(maxit):
        x1 = x0 - f(x0)/fp(x0)
        ea = abs((x1-x0)/x1)
        if ea < Ea:
            break
        x0 = x1
    return x1, f(x1), ea, i+1


