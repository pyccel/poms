# -*- coding: UTF-8 -*-
import utils
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix

# ... Compute Y = (B kron A) X
def kron_dot(B, A, X):
    # ...
    V = X.space
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    [s1, s2] = V.starts
    [e1, e2] = V.ends
    [p1, p2] = V.pads

    # ... For the results
    Y = StencilVector(V)
    Y[:,:] = 0.
    # ...

    # ... Temporary Vector
    X_tmp = StencilVector(V)
    # ...

    # ...
    X.update_ghost_regions()
    # ...

    # ..
    for j1 in range(s1-p1, e1+p1+1):
        for i2 in range(s2, e2+1):
             X_tmp[j1,i2] = np.dot( X[j1,i2-p2:i2+p2+1], B[i2,:])

    for i1 in range(s1, e1+1):
        for i2 in range(s2, e2+1):
             Y[i1,i2] = np.dot( A[i1,:], X_tmp[i1-p1:i1+p1+1,i2] )
    Y.update_ghost_regions()

    return Y

# ...

# ... Compute X, solution of (B kron A)X = Y
# ... Serial Version
def kron_solve_serial(B, A, Y):
    from scipy.linalg.lapack import dgetrf, dgetrs

    V = Y.space

    X = StencilVector(V)

    n1, n2 = V.npts
    p1, p2 = V.pads

    X_tmp  = np.zeros((n2, n1))

    A_arr = A.toarray()
    B_arr = B.toarray()

    A_lu, A_piv, A_finfo = dgetrf(A_arr)
    B_lu, B_piv, B_finfo = dgetrf(B_arr)

    for i2 in range(n2):
        X_tmp[i2, 0:n1], A_sinfo = dgetrs(A_lu, A_piv, Y[0:n1, i2])

    for i1 in range(n1):
        X[i1, 0:n2], B_infos = dgetrs(B_lu, B_piv, X_tmp[0:n2, i1])

    return X

# ...

# ... Compute X, solution of (B kron A)X = Y
# ... Parallel Version
def kron_solve_par(B, A, Y):
    from scipy.linalg.lapack import dgetrf, dgetrs

    # ...
    A_arr = A.toarray()
    B_arr = B.toarray()

    A_lu, A_piv, A_finfo = dgetrf(A_arr)
    B_lu, B_piv, B_finfo = dgetrf(B_arr)
    # ...

    # ...
    V = Y.space
    X = StencilVector(V)

    s1, s2 = V.starts
    e1, e2 = V.ends
    n1, n2 = V.npts

    subcomm_1 = V.cart.subcomm[0]
    subcomm_2 = V.cart.subcomm[1]
    # ...

    # ...
    Y_glob_1 = np.zeros((n1))

    Ytmp_glob_1 = np.zeros((n1, e2-s2+1))
    Ytmp_glob_2 = np.zeros((n2))

    X_glob_2 = np.zeros((e1-s1+1, n2))
    # ...


    # ...
    for i2 in range(e2-s2+1):
        Y_loc = Y[s1:e1+1, s2+i2].copy()
        subcomm_1.Allgatherv(Y_loc, Y_glob_1)

        Ytmp_glob_1[:,i2], A_sinfo = dgetrs(A_lu, A_piv, Y_glob_1)

    for i1 in range(e1-s1+1):
        Ytmp_loc = Ytmp_glob_1[s1+i1, 0:e2+1-s2].copy()
        subcomm_2.Allgatherv(Ytmp_loc, Ytmp_glob_2)

        X_glob_2[i1,:], B_sinfo = dgetrs(B_lu, B_piv, Ytmp_glob_2)
    # ...

    # ...
    X[s1:e1+1,s2:e2+1] = X_glob_2[:, s2:e2+1]

    return X
    # ...

# ...
