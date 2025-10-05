import numpy as np


def wegstein(g, x0, x1, Ea=1.e-7, maxit=30):
    """
    This function solves x=g(x) using the Wegstein method.
    The method is repeated until either the relative error
    falls below Ea (default 1.e-7) or reaches maxit (default 30).
    Input:
        g = name of the function for g(x)
        x0 = first initial guess for x
        x1 = second initial guess for x
        Ea = relative error threshold
        maxit = maximum number of iterations
    Output:
        x2 = solution estimate
        ea = relative error
        i+1 = number of iterations
    """
    for i in range(maxit):
        x2 = (x1*g(x0)-x0*g(x1))/(x1-x0-g(x1)+g(x0))
        ea = abs((x1-x0)/x1)
        if ea < Ea:
            break
        x0 = x1
        x1 = x2
    return x2, ea, i+1

