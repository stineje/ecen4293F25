import numpy as np


def bisect1(func, xl, xu, es=1.e-7, maxit=30):
    """
    Uses the bisection method to estimate a root of func(x).
    The method is iterated until the relative error from
    one iteration to the next falls below the specified
    value or until the maximum number of iterations is
    reached first.
    Input:
        func = name of the function
        xl = lower guess
        xu = upper guess
        es = relative error specification  (default 1.e-7)
        maxit = maximum number of iterations allowed (default 30)
    Output:
        xm = root estimate
        fm = function value at the root estimate
        ea = actual relative error achieved
        i+1 = number of iterations required
        or
        error message if initial guesses do not bracket solution
    """
    if func(xl)*func(xu) > 0:
        return 'initial estimates do not bracket solution'
    xmold = xl
    for i in range(maxit):
        xm = (xl+xu)/2
        ea = abs((xm-xmold)/xm)
        if ea < es:
            break
        if func(xm)*func(xl) > 0:
            xl = xm
        else:
            xu = xm
        xmold = xm
    return xm, func(xm), ea, i+1


def regfal(func, xl, xu, es=1.e-7, maxit=30):
    """
    Uses the false position method to estimate a root of func(x).
    The method is iterated until the relative error from
    one iteration to the next falls below the specified
    value or until the maximum number of iterations is
    reached first.
    Requirement: NumPy module must be imported
    Input:
        func = name of the function
        xl = lower guess
        xu = upper guess
        Ead = absolute error specification  (default 1.e-7)
        maxit = maximum number of iterations
     Output:
        xm = root estimate
        Ea = absolute error, last interval of uncertainty
        ea = actual relative error achieved
        n = number of iterations required
        or
        error message if initial guesses do not bracket solution
    """
    if func(xl)*func(xu) > 0:
        return 'initial estimates do not bracket solution'
    xmold = xl
    for i in range(maxit):
        xm = (func(xu)*xl-func(xl)*xu)/(func(xu)-func(xl))
        ea = abs((xm-xmold)/xm)
        if ea < es:
            break
        if func(xm)*func(xl) > 0:
            xl = xm
        else:
            xu = xm
        xmold = xm
    return xm, func(xm), ea, i+1


def gaussnaive(A, b):
    """
    gaussnaive: naive Gauss elimination
    input:
    A = coefficient matrix
    b = constant vector
    output:
    x = solution vector
    """
    (n, m) = A.shape
    # n = nm[0]
    # m = nm[1]
    if n != m:
        return 'Coefficient matrix A must be square'
    nb = n+1
    # build augmented matrix
    Aug = np.hstack((A, b))
    # forward elimination
    for k in range(n-1):
        for i in range(k+1, n):
            factor = Aug[i, k]/Aug[k, k]
            Aug[i, k:nb] = Aug[i, k:nb]-factor*Aug[k, k:nb]
    # back substitution
    x = np.zeros([n, 1])  # create empty x array
    x = np.matrix(x)  # convert to matrix type
    x[n-1] = Aug[n-1, nb-1]/Aug[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (Aug[i, nb-1]-Aug[i, i+1:n]*x[i+1:n, 0])/Aug[i, i]
    return x


def gausspivot(A, b):
    """
    gausspivot: Gauss elimination with partial pivoting
    input:
    A = coefficient matrix
    b = constant vector
    output:
    x = solution vector
    """
    (n, m) = A.shape
    if n != m:
        return 'Coefficient matrix A must be square'
    nb = n+1
    # build augmented matrix
    Aug = np.hstack((A, b))
    # forward elimination
    for k in range(n-1):

        # partial pivoting
        imax = maxrow(Aug[k:n, k])
        ipr = imax + k
        if ipr != k:  # no row swap if pivot is max
            for j in range(k, nb):  # swap rows k and ipr
                temp = Aug[k, j]
                Aug[k, j] = Aug[ipr, j]
                Aug[ipr, j] = temp

        for i in range(k+1, n):
            factor = Aug[i, k]/Aug[k, k]
            Aug[i, k:nb] = Aug[i, k:nb]-factor*Aug[k, k:nb]
    # back substitution
    x = np.zeros([n, 1])  # create empty x array
    x = np.matrix(x)  # convert to matrix type
    x[n-1] = Aug[n-1, nb-1]/Aug[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (Aug[i, nb-1]-Aug[i, i+1:n]*x[i+1:n, 0])/Aug[i, i]
    return x


def maxrow(avec):
    # function to determine the row index of the
    # maximum value in a vector
    maxrowind = 0
    n = len(avec)
    amax = abs(avec[0])
    for i in range(1, n):
        if abs(avec[i]) > amax:
            amax = avec[i]
            maxrowind = i
    return maxrowind


def tridiag(e, f, g, r):
    """
    tridiag: solves a set of n linear algebraic equations
             with a tridiagonal-banded coefficient matris
    input:
    e = subdiagonal vector of length n, first element = 0
    f = diagonal vector of length n
    g = superdiagonal vector of length n, last element = 0
    r = constant vector of length n
    output:
    x = solution vector of length n
    """
    n = len(f)
    # forward elimination
    x = np.zeros([n])
    for k in range(1, n):
        factor = e[k]/f[k-1]
        f[k] = f[k] - factor*g[k-1]
        r[k] = r[k] - factor*r[k-1]
    # back substitution
    x[n-1] = r[n-1]/f[n-1]
    for k in range(n-2, -1, -1):
        x[k] = (r[k] - g[k]*x[k+1])/f[k]
    return x
