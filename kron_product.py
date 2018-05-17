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
    for i1 in range(s1, e1+1):

        for i2 in range(s2, e2+1):

            for k1 in range(-p1, p1+1):
                j1 = i1 + k1
                X_tmp[j1, i2] = 0.

                for k2 in range(-p2, p2+1):
                    j2 = i2 + k2
                    X_tmp[j1, i2] += X[j1, j2]*B[i2, k2]

                X_tmp.update_ghost_regions(direction=1)

                Y[i1, i2] += A[i1, k1]*X_tmp[j1, i2]
            # ..
            Y.update_ghost_regions(direction=0)
            # ...
    # ...

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

    s1, s2 = V.starts
    e1, e2 = V.ends
    n1, n2 = V.npts

    subcomm_1 = V.cart._subcomm[0]
    subcomm_2 = V.cart._subcomm[1]
    # ...

    # ...
    X = StencilVector(V)

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
        Ytmp_loc = Ytmp_glob_1[s1+i1, s2:e2+1].copy()
        subcomm_2.Allgatherv(Ytmp_loc, Ytmp_glob_2)
        X_glob_2[i1,:], B_sinfo = dgetrs(B_lu, B_piv, Ytmp_glob_2)
    # ...

    # ...
    X[s1:e1+1,s2:e2+1] = X_glob_2[:, s2:e2+1]

    return X
    # ...

# ...

# ... TODO debug
def kron_dot_glob(B, A,  X):
    # ...
    V = X.space
    T = StencilVector(V)

    Y = StencilVector(V)
    Y[:,:] = 0.
    # ...

    # ..
    for i1 in range(V.npts[0]):

        i1_glob = i1 + V.starts[0]

        for i2 in range(V.npts[1]):
            i2_glob = i2 + V.starts[1]

            T[i1_glob, i2_glob] = 0.

            for k2 in range(-p2, p2+1):
                j2_glob = i2_glob + k2
                T[i1_glob, i2_glob] += X[i1_glob, j2_glob]*B[i2, k2]

            T.update_ghost_regions(direction=1)

            # ...
            for k1 in range(-p1, p1+1):
                j1_glob = i1_glob + k1
                Y[i1_glob, i2_glob] += A[i1, k1]*T[j1_glob, i2_glob]

            Y.update_ghost_regions(direction=0)
            # ...
    # ...

    return Y
# ...
