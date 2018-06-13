# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix
import utils
from kron_product       import kron_solve_serial, kron_solve_par
from spl.core.interface import collocation_cardinal_splines

'''
Test of solving: (B kron A) X = Y

To launch, run: mpirun -n 4 python3 tests/test_kron_03.py
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

    X_ref = utils.kron_solve_ref(A, B, Y)

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

# ... Parallel test
def test_par_glt(n1, n2, p1, p2):
    from utils import array_to_mat_stencil
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
    utils.populate_2d_vector(Y)

    C1 = collocation_cardinal_splines(p1, n1)
    C2 = collocation_cardinal_splines(p1, n1)
    M1 = array_to_mat_stencil(n1, p1, C1)
    M2 = array_to_mat_stencil(n2, p2, C2)
    # ...

    # ..
    wt = MPI.Wtime()
    X = kron_solve_par(M2, M1, Y)
    wt = MPI.Wtime() - wt

    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print('time: {}'.format(wt))
            print('Y  = \n', X.toarray())
            print('', flush=True)
        comm.Barrier()
    # ...


###############################################################################
if __name__ == '__main__':

    # ... numbers of elements and degres
    n1 = 4 ; n2 = 4
    p1 = 1 ; p2 = 1

    # ... serial test
    #test_ser(n1, n2, p1, p2)

    np.set_printoptions(linewidth=100000, precision=4)
    # ... parallel test
    test_par_glt(n1, n2, p1, p2)
