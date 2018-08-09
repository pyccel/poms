# -*- coding: UTF-8 -*-
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix
from kron_product       import kron_dot_v1, kron_dot_v2
import utils

'''
Serial and Parallel tests of kronnecker dot: Y = (B kron A) X
'''

# ... Serial tests
def test_ser(n1, n2, p1, p2):
    # ... Vector Spaces
    V = StencilVectorSpace([n1, n2], [p1, p2], [False, False])
    V1 = StencilVectorSpace([n1], [p1], [False])
    V2 = StencilVectorSpace([n2], [p2], [False])

    # ... Inputs
    X = StencilVector(V)
    A = StencilMatrix(V1, V1)
    B = StencilMatrix(V2, V2)
    # ...

    # ... Fill in A, B and X
    utils.populate_1d_matrix(A, 5.)
    utils.populate_1d_matrix(B, 6.)
    utils.populate_2d_vector(X)
    # ...

    # ...
    Y_ref = utils.kron_dot_ref(A, B, X)
    Y1 = kron_dot_v1(A, B, X)
    Y2 = kron_dot_v2(A, B, X)

    print('Y_ref = \n', Y_ref)
    print('Y1    = \n', Y1.toarray())
    print('Y2    = \n', Y2.toarray())
    # ...
# ...

# ... Parallel test (1st version)
def test_par_v1(n1, n2, p1, p2):
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
    X = StencilVector(V)
    A = StencilMatrix(V1, V1)
    B = StencilMatrix(V2, V2)
    # ...

    # ... Fill in A, B and X
    utils.populate_1d_matrix(A, 5.)
    utils.populate_1d_matrix(B, 6.)
    utils.populate_2d_vector(X)
    # ...

    # ..
    Y = kron_dot_v1(A, B, X)
    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print('Y  = \n', Y.toarray())
            print('', flush=True)
        comm.Barrier()
    # ...

# ... Parallel test (2nd version)
def test_par_v2(n1, n2, p1, p2):
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
    X = StencilVector(V)
    A = StencilMatrix(V1, V1)
    B = StencilMatrix(V2, V2)
    # ...

    # ... Fill in A, B and X
    utils.populate_1d_matrix(A, 5.)
    utils.populate_1d_matrix(B, 6.)
    utils.populate_2d_vector(X)
    # ...

    # ..
    Y = kron_dot_v2(A, B, X)
    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print('Y  = \n', Y.toarray())
            print('', flush=True)
        comm.Barrier()
    # ...


###############################################################################
if __name__ == '__main__':


    # ... numbers of elements and degres
    n1 = 8 ; n2 = 4
    p1 = 2 ; p2 = 1

    # ... serial test
    #test_ser(n1, n2, p1, p2)

    # ... parallel test
    # To launch an example, run: mpirun -n 2 python3 tests/test_kron_dot.py
    #test_par_v1(n1, n2, p1, p2)
    test_par_v2(n1, n2, p1, p2)
