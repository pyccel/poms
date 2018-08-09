# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix
from kron_product       import kron_solve_serial, kron_solve_par
from spl.core.interface import collocation_cardinal_splines

'''
Test of solving: (B kron A) X = Y

To launch, run: mpirun -n 4 python3 tests/test_kron_solve.py
'''

# .. Fill in stencil matrix as band
def populate_1d_matrix(M, diag):
    e = M.ends[0]
    s = M.starts[0]
    p = M.pads[0]

    for i in range(s, e+1):
        for k in range(-p, p+1):
            M[i, k] = k
        M[i, 0] = diag
    M.remove_spurious_entries()
# ...

# .. Fill in stencil vector
def populate_2d_vector(X):
    e1 = X.ends[0]
    s1 = X.starts[0]
    e2 = X.ends[1]
    s2 = X.starts[1]

    for i1 in range(s1, e1+1 ):
        for i2 in range(s2, e2+1):
            X[i1,i2] = 1.
# ...

# ... reference: return X, solution of (B kron A)X = Y
def kron_solve_ref(B, A, Y):
    from numpy import zeros
    from scipy.sparse import csc_matrix, kron
    from scipy.sparse.linalg import splu

    # ...
    A_csr = A.tocsr()
    B_csr = B.tocsr()
    C = csc_matrix(kron(B_csr, A_csr))

    # ...
    V = Y.space

    [s1, s2] = V.starts
    [e1, e2] = V.ends
    [p1, p2] = V.pads

    # ...
    Y_arr = Y.toarray()
    C_op  = splu(C)
    X = C_op.solve(Y_arr)

    return X
# ...

# ... serial test
def test_ser(n1, n2, p1, p2):
    # ... Vector Spaces
    V  = StencilVectorSpace([n1, n2], [p1, p2], [False, False])
    V1 = StencilVectorSpace([n1], [p1], [False])
    V2 = StencilVectorSpace([n2], [p2], [False])

    # ... Inputs
    Y = StencilVector(V)
    A = StencilMatrix(V1, V1)
    B = StencilMatrix(V2, V2)
    # ...

    # ... Fill in A, B and X
    populate_1d_matrix(A, 5.)
    populate_1d_matrix(B, 6.)
    populate_2d_vector(Y)
    # ...

    # ..
    X = kron_solve_serial(B, A, Y)

    X_ref = kron_solve_ref(A, B, Y)

    print('X =  \n', X.toarray())
    print('X_ref =  \n', X_ref)
    # ...
# ...

# ... parallel test
def test_par(n1, n2, p1, p2):
    # ...
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # ...

    # ... 2D MPI cart
    cart = Cart(npts = [n1, n2], pads = [p1, p2], periods = [False, False],\
                reorder = True, comm = comm)

    # ... Vector Spaces
    V = StencilVectorSpace(cart)
    V1 = StencilVectorSpace([n1], [p1], [False])
    V2 = StencilVectorSpace([n2], [p2], [False])

    # ... Inputs
    Y = StencilVector(V)
    A = StencilMatrix(V1, V1)
    B = StencilMatrix(V2, V2)
    # ...

    # ... Fill in A, B and X
    populate_1d_matrix(A, 5.)
    populate_1d_matrix(B, 6.)
    populate_2d_vector(Y)
    # ...

    # ..
    X = kron_solve_par(B, A, Y)

    for i in range(comm.Get_size()):
        #if rank == i:
           # print('rank= ', rank)
          #  print('Y  = \n', X.toarray())
           # print('', flush=True)
        comm.Barrier()
    # ...


###############################################################################
if __name__ == '__main__':

    # ... numbers of elements and degres
    n1 = 150 ; n2 = 150
    p1 = 10 ; p2 = 10

    # ... serial test
    #test_ser(n1, n2, p1, p2)

    np.set_printoptions(linewidth=100000, precision=4)
    # ... parallel test
    test_par(n1, n2, p1, p2)
