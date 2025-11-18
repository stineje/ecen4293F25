import numpy as np

def gaussnaive(A,b):
    """
    gaussnaive: naive Gauss elimination
    input:
    A = coefficient matrix
    b = constant vector
    output:
    x = solution vector
    """
    (n,m) = A.shape
    #n = nm[0]
    #m = nm[1]
    if n != m:
        return 'Coefficient matrix A must be square'
    nb = n+1
    # build augmented matrix
    Aug = np.hstack((A,b))
       # forward elimination
    for k in range(n-1):
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

A = np.matrix('0.55,0.25,0.25 ; 0.30,0.45,0.20 ; 0.15,0.30,0.55')
b = np.matrix('4800;5800;5700')
V = gaussnaive(A,b)
print('Volume from Pit 1:',V[0])
print('Volume from Pit 2:',V[1])
print('Volume from Pit 3:',V[2])

