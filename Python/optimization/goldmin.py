import numpy as np


def goldmin(f, xl, xu, Ea=1.e-7, maxit=30):
    """
    use the golden-section search to find the minimum of f(x)
    input:
        f = name of the function
        xl = lower initial guess
        xu = upper initial guess
        Ea = absolute relative error criterion (default = 1.e-7)
        maxit = maximum number of iterations (default = 30)
    output:
        xopt = location of the minimum
        f(xopt) = function value at the minimum
        ea = absolute relative error achieved
        i+1 = number of iterations required
    """
    phi = (1+np.sqrt(5))/2
    d = (phi - 1)*(xu-xl)
    x1 = xl + d
    f1 = f(x1)
    x2 = xu - d
    f2 = f(x2)
    for i in range(maxit):
        xint = xu - xl
        if f1 < f2:
            xopt = x1
            xl = x2
            x2 = x1
            f2 = f1
            x1 = xl + (phi-1)*(xu-xl)
            f1 = f(x1)
        else:
            xopt = x2
            xu = x1
            x1 = x2
            f1 = f2
            x2 = xu - (phi-1)*(xu-xl)
            f2 = f(x2)
        if xopt != 0:
            ea = (2-phi)*abs(xint/xopt)
            if ea <= Ea:
                break
    return xopt, f(xopt), ea, i+1
