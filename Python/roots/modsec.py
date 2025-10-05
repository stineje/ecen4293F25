import numpy as np


def modsec(f, x0, delta=1.e-5, Ea=1.e-7, maxit=30):
    """
    function to solve for the root of f(x) using the secant method
    inputs:
        f = name of f(x) function
        x0 = initial guess
        delta = perturbation fraction (default 1.e-5)
        Ea = relative error criterion (default 1.e-7)
        maxit = maximum number of iterations (default 30)
    outputs:
        x2 = solution estimate for x
        f(x2) = function value at the solution estimate
        ea = relative error achieved
        i+1 = number of iterations taken
    """
    for i in range(maxit):
        x1 = x0-f(x0)/(f((1+delta)*x0)-f(x0))*delta*x0
        ea = abs((x1-x0)/x1)
        if ea < Ea:
            break
        x0 = x1
    return x1, f(x1), ea, i+1
