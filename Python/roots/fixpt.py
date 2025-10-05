import numpy as np


def fixpt(g, x0, Ea=1.e-7, maxit=30):
    """
    This function solves x=g(x) using fixed-point iteration.
    The method is repeated until either the relative error
    falls below Ea (default 1.e-7) or reaches maxit (default 30).
    Input:
        g = name of the function for g(x)
        x0 = initial guess for x
        Ea = relative error threshold
        maxit = maximum number of iterations
    Output:
        x1 = solution estimate
        ea = relative error
        i+1 = number of iterations
    """
    for i in range(maxit):
        x1 = g(x0)
        ea = abs((x1-x0)/x1)
        print(x1,ea)
        if ea < Ea:
            break
        x0 = x1
    return x1, ea, i+1


