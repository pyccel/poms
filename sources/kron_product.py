# -*- coding: UTF-8 -*-
import utils
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix

# ... Compute Y = (A kron B) X (1st parallel version)
def kron_dot_v1(A, B, X):
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

# ... Compute Y = (A kron B) X (2nd parallel version)
def kron_dot_v2(A, B, X):
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

# ... Compute X, solution of (A kron B)X = Y (Serial Version)
def kron_solve_serial(A, B, Y):
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

# ... Compute X, solution of (A kron B)X = Y (Parallel Version)
def kron_solve_par(A, B, Y):
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

# ... convert a 1D stencil matrix to band matrix
def to_bnd(A):

    dmat = dia_matrix(A.toarray())
    la    = abs(dmat.offsets.min())
    ua    = dmat.offsets.max()
    cmat = dmat.tocsr()

    A_bnd = np.zeros((1+ua+2*la, cmat.shape[1]))

    for i,j in zip(*cmat.nonzero()):
        A_bnd[la+ua+i-j, j] = cmat[i,j]

    return A_bnd, la, ua
# ...

# ...
def kron_solve_bnd_par(A, B, Y):
    from scipy.linalg.lapack import dgbtrs

    # ...
    A_bnd = A[0]; la= A[1]; ua = A[2]; A_piv = A[3]
    B_bnd = B[0]; lb= B[1]; ub = B[2]; B_piv = B[3]
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

    lwt = MPI.Wtime()
    # ...
    for i2 in range(e2-s2+1):
        Y_loc = Y[s1:e1+1, s2+i2].copy()
        subcomm_1.Allgatherv(Y_loc, Y_glob_1)

        Ytmp_glob_1[:,i2], A_sinfo = dgbtrs(A_bnd, la, ua, Y_glob_1, A_piv)

    for i1 in range(e1-s1+1):
        Ytmp_loc = Ytmp_glob_1[s1+i1, 0:e2+1-s2].copy()
        subcomm_2.Allgatherv(Ytmp_loc, Ytmp_glob_2)

        X_glob_2[i1,:], B_sinfo = dgbtrs(B_bnd, lb, ub, Ytmp_glob_2, B_piv)
    # ...

    # ...
    X[s1:e1+1,s2:e2+1] = X_glob_2[:, s2:e2+1]
    lwt = MPI.Wtime()  - lwt
    return X, lwt
    # ...
# ...
