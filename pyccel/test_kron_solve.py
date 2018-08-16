# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix
from kron_product       import kron_solve_serial, kron_solve_par,kron_solve_par_bnd_2d,kron_solve_par_bnd_3d
from spl.core.interface import collocation_cardinal_splines
from scipy.sparse import dia_matrix,csc_matrix
from scipy.sparse.linalg import splu
from scipy import kron
from scipy.linalg import solve

'''
Test of solving: (A kron B) X = Y

To launch, run: mpirun -n 4 python3 tests/test_kron_solve.py
'''
def kron_solve_seq_ref_2d(A1, A2, Y):

    C  = kron(A1, A2)
    Y_ = Y.flatten()
    X  = solve(C, Y_)
    
    return X.reshape(Y.shape)

def kron_solve_seq_ref_3d(A1, A2, A3,Y):

    C  = kron(kron(A1, A2), A3)
    Y_ = Y.flatten()
    X  = solve(C, Y_)
    
    return X.reshape(Y.shape)


def to_bnd(A):
    dmat = dia_matrix( A )
    la = abs( dmat.offsets.min() )
    ua = dmat.offsets.max()
    cmat = dmat.tocsr()
    A_bnd = np.zeros( (1+ua+2*la, cmat.shape[1]),order='F' )
    for i,j in zip( *cmat.nonzero() ):
        A_bnd[la+ua+i-j,j] = cmat[i,j]
    return A_bnd ,la, ua

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

def test_par_banded_2d(n1, n2, p1, p2 ,P1=False, P2=False):
    # ...
    

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ... 2D MPI cart
    cart = Cart(npts = [n1, n2], pads = [p1, p2], periods = [P1, P2],\
                reorder = True, comm = comm)

    V = StencilVectorSpace(cart)

    [s1, s2] = V.starts
    [e1, e2] = V.ends
    
    V1 = StencilVectorSpace([n1], [p1], [P1])
    V2 = StencilVectorSpace([n2], [p2], [P2])

    # ... Matrices and Direct solvers
    A1 = StencilMatrix(V1, V1)
    A1[:,-p1:0   ] = -4
    A1[:, 0 :1   ] = 10*p1
    A1[:, 1 :p1+1] = -2
    A1.remove_spurious_entries()
    

    A2 = StencilMatrix(V2, V2)
    A2[:,-p2:0   ] = -1
    A2[:, 0 :1   ] = 2*p2
    A2[:, 1 :p2+1] = -2
    A2.remove_spurious_entries()
   

    #  ... RHS
    Y = StencilVector(V)
    Y_glob = np.array([[(i1+1)*10+(i2+1) for i2 in range(n2)] for i1 in range(n1)])
    Y[s1:e1+1, s2:e2+1] = Y_glob[s1:e1+1, s2:e2+1]
    Y.update_ghost_regions()

    A1 = A1.toarray()
    A2 = A2.toarray()
    
    
    

    X_glob = kron_solve_seq_ref_2d(A1, A2, Y_glob)
   
    A1_bnd, la1, ua1 = to_bnd(A1)
    A2_bnd, la2, ua2 = to_bnd(A2)
   
    X = StencilVector(V)
    X._data = X._data.copy(order = 'F')
    Y._data = Y._data.copy(order = 'F')

    wt = MPI.Wtime()
    kron_solve_par_bnd_2d(A1_bnd,la1 ,ua1 ,A2_bnd, la2, ua2, Y, X)
    wt = MPI.Wtime() - wt
    
    #import time 
    
    #for i in range(comm.Get_size()):
    #    if rank == i:
    #        print('rank= ', rank)
    #        print('X_glob  = \n', X_glob)
    #        print('X  = \n', X.toarray().reshape(n1,n2))
    #        print('', flush=True)
    #        time.sleep(0.1)
    #    comm.Barrier()
    assert np.allclose( X[s1:e1+1, s2:e2+1], X_glob[s1:e1+1, s2:e2+1])
        
    
    
    # ...

def test_kron_solver_3d_par( n1, n2, n3, p1, p2, p3, P1=False, P2=False, P3=False ):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ... 2D MPI cart
    cart = Cart(npts = [n1, n2, n3], pads = [p1, p2, p3], periods = [P1, P2, P3],\
                reorder = True, comm = comm)

    # ...
    sizes1 = cart.global_ends[0] - cart.global_starts[0] + 1
    sizes2 = cart.global_ends[1] - cart.global_starts[1] + 1
    sizes3 = cart.global_ends[2] - cart.global_starts[2] + 1
    # ...

    # ... Vector Spaces
    V = StencilVectorSpace(cart)

    [s1, s2, s3] = V.starts
    [e1, e2, e3] = V.ends


   # TODO: make MPI type available through property
    mpi_type = V._mpi_type
    # ...

    V1 = StencilVectorSpace([n1], [p1], [P1])
    V2 = StencilVectorSpace([n2], [p2], [P2])
    V3 = StencilVectorSpace([n3], [p3], [P3])

    # ... Matrices and Direct solvers
    A1 = StencilMatrix(V1, V1)
    A1[:,-p1:0   ] = -4
    A1[:, 0 :1   ] = 10*p1
    A1[:, 1 :p1+1] = -4
    A1.remove_spurious_entries()
    

    A2 = StencilMatrix(V2, V2)
    A2[:,-p2:0   ] = -1
    A2[:, 0 :1   ] = 2*p2
    A2[:, 1 :p2+1] = -1
    A2.remove_spurious_entries()
    

    A3 = StencilMatrix(V3, V3)
    A3[:,-p3:0   ] = -2
    A3[:, 0 :1   ] = 3*p2
    A3[:, 1 :p3+1] = -2
    A3.remove_spurious_entries()
    

    #  ... RHS
    Y = StencilVector(V)
    Y_glob = np.array([[[(i1+1)*100+(i2+1)*10 +(i3+1) for i3 in range(n3)] for i2 in range(n2)] for i1 in range(n1)])
    Y[s1:e1+1, s2:e2+1, s3:e3+1] = Y_glob[s1:e1+1, s2:e2+1, s3:e3+1]
    Y.update_ghost_regions()

    # ...

    A1 = A1.toarray()
    A2 = A2.toarray()
    A3 = A3.toarray()
  
    A1_bnd, la1, ua1 = to_bnd(A1)
    A2_bnd, la2, ua2 = to_bnd(A2)
    A3_bnd, la3, ua3 = to_bnd(A3)
   
    X = StencilVector(V)
    X._data = X._data.copy(order = 'F')
    Y._data = Y._data.copy(order = 'F')
    X_glob = kron_solve_seq_ref_3d(A1, A2, A3, Y_glob)
    

    wt = MPI.Wtime()
    kron_solve_par_bnd_3d(A1_bnd, la1 , ua1 , A2_bnd, la2, ua2, A3_bnd, la3, ua3, Y, X)
    wt = MPI.Wtime() - wt
        
    #import time 
    
    #for i in range(comm.Get_size()):
    #    if rank == i:
    #        print('rank= ', rank, 'time=',wt)
    #        print('X_glob  = \n', X_glob)
    #        print('X  = \n', X.toarray().reshape(n1,n2,n3))
    #        print('', flush=True)
    #        time.sleep(0.1)
    #    comm.Barrier()
    assert np.allclose( X[s1:e1+1, s2:e2+1, s3:e3+1], X_glob[s1:e1+1, s2:e2+1, s3:e3+1] )
#

###############################################################################
if __name__ == '__main__':

    # ... numbers of elements and degres


    n1 = 10	 ; n2 = 10 ; n3 = 10	
    p1 = 1   ; p2 = 1 ; p3 = 1

    # ... serial test
    #test_ser(n1, n2, p1, p2)
    #test_par(n1, n2, p1, p2)
    test_par_banded_2d(n1, n2, p1, p2)
    test_kron_solver_3d_par(n1, n2, n3, p1, p2, p3)


