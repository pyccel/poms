# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix
from kron_product       import kron_solve_serial, kron_solve_par,kron_solve_par_bnd
from spl.core.interface import collocation_cardinal_splines
from scipy.sparse import dia_matrix

'''
Test of solving: (A kron B) X = Y

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

    A = A.toarray().copy(order = 'F')
    B = B.toarray().copy(order = 'F')

    X = kron_solve_par(A, B, Y)

    for i in range(comm.Get_size()):
        #if rank == i:
           # print('rank= ', rank)
          #  print('Y  = \n', X.toarray())
           # print('', flush=True)
        comm.Barrier()

    wt = MPI.Wtime()
    X  = kron_solve_par(B, A, Y)
    wt = MPI.Wtime() - wt

    print('rank: ', rank, '- elapsed time: {}'.format(wt))

    # ...

def test_par_banded(n1, n2, p1, p2):
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

    A = A.toarray()
    B = B.toarray()
    dmat = dia_matrix( A )
    la = abs( dmat.offsets.min() )
    ua = dmat.offsets.max()
    cmat = dmat.tocsr()
    A_bnd = np.zeros( (1+ua+la, cmat.shape[1]),order='F' )
    for i,j in zip( *cmat.nonzero() ):
        A_bnd[ua+i-j,j] = cmat[i,j]

    dmat = dia_matrix( B )
    lb = abs( dmat.offsets.min() )
    ub = dmat.offsets.max()
    cmat = dmat.tocsr()
    B_bnd = np.zeros( (1+ub+lb, cmat.shape[1]),order='F' )
    for i,j in zip( *cmat.nonzero() ):
        B_bnd[ub+i-j,j] = cmat[i,j]
    #print(2*la+ua+1)
    wt = MPI.Wtime()
    X  = kron_solve_par_bnd(A_bnd,la ,ua ,B_bnd, lb, ub, Y)
    wt = MPI.Wtime() - wt

    print('rank: ', rank, '- elapsed time: {}'.format(wt))

    # ...


###############################################################################
if __name__ == '__main__':

    # ... numbers of elements and degres


    n1 = 50 ; n2 = 10
    p1 = 3 ; p2 = 3

    # ... serial test
    #test_ser(n1, n2, p1, p2)
    #test_par(n1, n2, p1, p2)
    test_par_banded(n1, n2, p1, p2)


