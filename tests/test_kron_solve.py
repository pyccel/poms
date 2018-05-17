# -*- coding: UTF-8 -*-

from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix
import utils
from kron_product       import kron_solve_serial, kron_solve_par

'''
Test of solving: (B kron A) X = Y

To launch, run: python3 tests/test_kron_03.py
'''

# ... Serial test
def test_ser(n1, n2, p1, p2):
    # ... Vector Spaces
    V = StencilVectorSpace([n1, n2], [p1, p2], [False, False])
    V1 = StencilVectorSpace([n1], [p1], [False])
    V2 = StencilVectorSpace([n2], [p2], [False])

    # ... Inputs
    Y = StencilVector(V)
    A = StencilMatrix(V1, V1)
    B = StencilMatrix(V2, V2)
    # ...

    # ... Fill in A, B and X
    utils.populate_1d_matrix(A, 5.)
    utils.populate_1d_matrix(B, 6.)
    utils.populate_2d_vector(Y)
    # ...

    # ..
    X = kron_solve_serial(B, A, Y)

    X_ref = utils.kron_solve_ref(B, A, Y)

    print('X =  \n', X.toarray())
    print('X_ref =  \n', X_ref)
    # ...
# ...


# ... Parallel test
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
    utils.populate_1d_matrix(A, 5.)
    utils.populate_1d_matrix(B, 6.)
    utils.populate_2d_vector(Y)
    # ...

    # ..
    X = kron_solve_par(B, A, Y)

    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print('Y  = \n', X.toarray())
            print('', flush=True)
        comm.Barrier()
    # ...

###############################################################################
if __name__ == '__main__':

    # ... numbers of elements and degres
    n1 = 8 ; n2 = 4
    p1 = 1 ; p2 = 1

    # ... serial test
#    test_ser(n1, n2, p1, p2)

    # ... parallel test
    test_par(n1, n2, p1, p2)
