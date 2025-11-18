import numpy as np

def gausspivot(A,b):
    """
    gausspivot: Gauss elimination with partial pivoting
    input:
    A = coefficient matrix
    b = constant vector
    output:
    x = solution vector
    """
    (n,m) = A.shape
    if n != m:
        return 'Coefficient matrix A must be square'
    nb = n+1
    # build augmented matrix
    Aug = np.hstack((A,b))
       # forward elimination
    for k in range(n-1):

        # partial pivoting
        imax = maxrow(Aug[k:n,k])
        ipr = imax + k
        if ipr != k:  # no row swap if pivot is max
            for j in range(k,nb):  # swap rows k and ipr
                temp = Aug[k,j]
                Aug[k,j] = Aug[ipr,j]
                Aug[ipr,j] = temp

        for i in range(k+1,n):
            factor = Aug[i,k]/Aug[k,k]
            Aug[i,k:nb]=Aug[i,k:nb]-factor*Aug[k,k:nb]
    # back substitution
    x = np.zeros([n,1])  # create empty x array
    x = np.matrix(x)  # convert to matrix type
    x[n-1]=Aug[n-1,nb-1]/Aug[n-1,n-1]
    for i in range(n-2,-1,-1):
        x[i]=(Aug[i,nb-1]-Aug[i,i+1:n]*x[i+1:n,0])/Aug[i,i]
    return x

def maxrow(avec):
    # function to determine the row index of the
    # maximum value in a vector
    maxrowind = 0
    n = len(avec)
    amax = abs(avec[0])
    for i in range(1,n):
        if abs(avec[i]) > amax:
            amax = avec[i]
            maxrowind = i
    return maxrowind
   
    
A = np.matrix('0 -0.1 -0.2 ; 0.1 7 -0.3 ; 0.3 -0.2 10')
b = np.matrix(' 7.85 ; -19.3 ; 71.4')
x = gausspivot(A,b)
print(x)
xtest = np.linalg.inv(A)*b
print(xtest)
