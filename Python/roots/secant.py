import numpy as np


def secant(f, x0, x1, Ea=1.e-7, maxit=30):
    """
    function to solve for the root of f(x) using the secant method
    inputs:
        f = name of f(x) function
        x0 = initial guess
        x1 = initial guess
        Ea = relative error criterion
        maxit = maximum number of iterations
    outputs:
        x2 = solution estimate for x
        f(x2) = function value at the solution estimate
        ea = relative error achieved
        i+1 = number of iterations taken
    """
    for i in range(maxit):
        x2 = x1-f(x1)/(f(x1)-f(x0))*(x1-x0)
        ea = abs((x2-x1)/x2)
        if ea < Ea:
            break
        x0 = x1
        x1 = x2
    return x2, f(x2), ea, i+1


