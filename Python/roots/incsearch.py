import numpy as np

def incsearch(func, xmin, xmax, ns=50):
    """
    incsearch: incremental search locator
        incsearch(func,xmin,xmax,ns)
        finds brackets of x that contain sign changes in
        a function of x on an interval
    input: 
        func = name of the function
        xmin, xmax = endpoints of the interval
        ns = number of subintervals, default value = 50
    output:  a tuple containing
        nb = number of bracket pairs found
        xb = list of bracket pair values
        or returns "no brackets found"
    """
    x = np.linspace(xmin, xmax, ns)  # create array of x values
    f = []  # build array of corresponding function values
    for k in range(ns-1):
        f.append(func(x[k]))
    nb = 0
    xb = []
    for k in range(ns-2):  # check adjacent pairs of function values
        if func(x[k])*func(x[k+1]) < 0:  # for sign change
            nb = nb + 1  # increment the bracket counter
            xb.append((x[k], x[k+1]))  # save the bracketing pair
    if nb == 0:
        return 'no brackets found'
    else:
        return nb, xb


