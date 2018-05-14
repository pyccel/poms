# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix

# .. Fill in stencil matrix as band
def populate_1d_matrix(M, diag):
    e = M.ends[0]
    s = M.starts[0]
    p = M.pads[0]

    for i in range(s, e+1):
#        for k in range(-p, p+1):
#            M[i, k] = k
        M[i, 0] = diag
    M.remove_spurious_entries()
# ...

def populate_2d_vector(X):
    e1 = X.ends[0]
    s1 = X.starts[0]
    e2 = X.ends[1]
    s2 = X.starts[1]

    for i1 in range(s1, e1+1 ):
        for i2 in range(s2, e2+1):
            X[i1,i2] = i1 + i2 + 1
# ...

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

# ... For testing
def kron_dot_scipy(B, A, X):
    from scipy import sparse
    import numpy as np

    # ...
    A_csr = A.tocsr()
    B_csr = B.tocsr()
    C = sparse.kron(B_csr, A_csr)

    # ...
    V = X.space

    [s1, s2] = V.starts
    [e1, e2] = V.ends
    [p1, p2] = V.pads

    X_arr = np.zeros((e1+1-s1)*(e2+1-s2))
    i = 0
    for i2 in range(s2, e2+1):
        for i1 in range(s1, e1+1):
            X_arr[i] = X[i1, i2]
            i += 1
    # ...

    Y_arr = C.dot(X_arr)

    return C, X_arr, Y_arr
# ...

# ...
def test_par(n1, n2, p1, p2):
    # ...
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ... 2D MPI cart
    cart = Cart(npts=[n1, n2], pads=[p1, p2], periods=[False, False], reorder=True, comm=comm)

    # ... Vector Spaces
    V = StencilVectorSpace(cart)
    V1 = StencilVectorSpace([n1], [p1], [False])
    V2 = StencilVectorSpace([n2], [p2], [False])

    # ...
    X = StencilVector(V)
    A = StencilMatrix(V1, V1)
    B = StencilMatrix(V2, V2)
    # ...


    populate_1d_matrix(A, 5.)
    populate_1d_matrix(B, 6.)
    populate_2d_vector(X)
    X.update_ghost_regions()

    Y = kron_dot(B, A, X)

    # ...
    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
#            print('A =  \n', A.toarray())
#            print('B =  \n', B.toarray())
#            print('X = \n', X._data)
            print('Y  = \n', Y._data)
            print('', flush=True)
        comm.Barrier()
    # ...
# ...


def test_seq(n1, n2, p1, p2):
    # ... Vector Spaces
    V = StencilVectorSpace([n1, n2], [p1, p2], [False, False])
    V1 = StencilVectorSpace([n1], [p1], [False])
    V2 = StencilVectorSpace([n2], [p2], [False])

    # ...
    X = StencilVector(V)
    A = StencilMatrix(V1, V1)
    B = StencilMatrix(V2, V2)
    # ...


    populate_1d_matrix(A, 5.)
    populate_1d_matrix(B, 6.)
    populate_2d_vector(X)

    Y = kron_dot(B, A, X)

    C, Xs, Ys = kron_dot_scipy(B, A, X)

    print('A =  \n', A.toarray())
    print('B =  \n', B.toarray())
    print('C = \n', C.toarray())
    print('X = \n', Xs)
    print('Y  = \n', Y._data)
    print('Y_ref = \n', Ys)
    # ...
# ...

#===============================================================================
if __name__ == "__main__":

    # ... The discretization
    n1 = 8
    n2 = 3
    p1 = 1
    p2 = 1

    # ..
#    test_seq(n1, n2, p1, p2)

    test_par(n1, n2, p1, p2)
