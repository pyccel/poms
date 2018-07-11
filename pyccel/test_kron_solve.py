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
    wt = MPI.Wtime()
    X  = kron_solve_par(B, A, Y)
    wt = MPI.Wtime() - wt

    print('rank: ', rank, '- elapsed time: {}'.format(wt))
    # ...


###############################################################################
if __name__ == '__main__':

    # ... numbers of elements and degres
    n1 = 8 ; n2 = 8
    p1 = 3 ; p2 = 3

    test_par(n1, n2, p1, p2)
