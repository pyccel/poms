# -*- coding: UTF-8 -*-
import utils
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

    T = StencilVector(V)

    Y = StencilVector(V)
    Y[:,:] = 0.
    # ...

    # ..
    for i1 in range(s1, e1+1):

        for i2 in range(s2, e2+1):

            for k1 in range(-p1, p1+1):
                j1 = i1 + k1
                T[j1, i2] = 0.

                for k2 in range(-p2, p2+1):
                    j2 = i2 + k2
                    T[j1, i2] += X[j1, j2]*B[i2, k2]

                T.update_ghost_regions(direction=1)

                Y[i1, i2] += A[i1, k1]*T[j1, i2]
            # ..
            Y.update_ghost_regions(direction=0)
            # ...
    # ...

    return Y

# ...

# ... Compute X, solution of (B kron A)X = Y
def kron_solve(B, A, Y):
    from scipy.linalg.lapack import dgetrf, dgetrs

    V = Y.space

    X = StencilVector(V)

    [s1, s2] = V.starts
    [e1, e2] = V.ends
    [p1, p2] = V.pads

    n1 = e1 - s1 + 1
    n2 = e2 - s2 + 1

    Vt = StencilVectorSpace([n2, n1], [p2, p1], [False, False])
    Xt = StencilVector(Vt)

    # A is n1xn1 matrix
    # B is n2xn2 matrix

    A_arr = A.toarray()
    B_arr = B.toarray()

    A_lu, A_piv, A_finfo = dgetrf(A_arr)
    B_lu, B_piv, B_finfo = dgetrf(B_arr)

    for i2 in range(n2):
        Xt[i2, 0:n1], A_sinfo = dgetrs(A_lu, A_piv, Y[0:n1, i2])

    for i1 in range(n1):
        X[i1, 0:n2], B_infos = dgetrs(B_lu, B_piv, Xt[0:n2, i1])

    return X

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
