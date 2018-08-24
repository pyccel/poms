# -*- coding: UTF-8 -*-

import pytest
import time
import numpy as np
from mpi4py                     import MPI
from spl.ddm.cart               import Cart
from scipy.sparse               import csc_matrix, dia_matrix, kron
from scipy.sparse.linalg        import splu
from spl.linalg.stencil         import StencilVectorSpace, StencilVector, StencilMatrix

from kron_product            import kron_solve_bnd_par


# ... return X, solution of (A1 kron A2)X = Y
def kron_solve_seq_ref(A1, A2, Y):

    # ...
    A1_csr = A1.tocsr()
    A2_csr = A2.tocsr()
    C = csc_matrix(kron(A1_csr, A2_csr))

    C_op  = splu(C)
    X = C_op.solve(Y.flatten())

    return X.reshape(Y.shape)
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

#===============================================================================
def test_kron_solver_2d_band_par( n1, n2, p1, p2, P1=False, P2=False ):

    from scipy.linalg.lapack import dgbtrf
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ... 2D MPI cart
    cart = Cart(npts = [n1, n2], pads = [p1, p2], periods = [P1, P2],\
                reorder = True, comm = comm)

    # ... Vector Spaces
    V = StencilVectorSpace(cart)

    [s1, s2] = V.starts
    [e1, e2] = V.ends


   # TODO: make MPI type available through property
    mpi_type = V._mpi_type
    # ...

    V1 = StencilVectorSpace([n1], [p1], [P1])
    V2 = StencilVectorSpace([n2], [p2], [P2])

    # ... Matrices and Direct solvers
    A1 = StencilMatrix(V1, V1)
    A1[:,-p1:0   ] = -4
    A1[:, 0 :1   ] = 10*p1
    A1[:, 1 :p1+1] = -4
    A1.remove_spurious_entries()
    A1_bnd, la1, ua1 = to_bnd(A1)
    A1_bnd, A1_piv, A1_finfo = dgbtrf(A1_bnd, la1, ua1)
    # ...

    # ...
    A2 = StencilMatrix(V2, V2)
    A2[:,-p2:0   ] = -1
    A2[:, 0 :1   ] = 2*p2
    A2[:, 1 :p2+1] = -1
    A2.remove_spurious_entries()
    A2_bnd, la2, ua2 = to_bnd(A2)
    A2_bnd, A2_piv, A2_finfo = dgbtrf(A2_bnd, la2, ua2)
    # ...

    #  ... RHS
    Y = StencilVector(V)
    Y_glob = np.array([[(i1+1)*10+(i2+1) for i2 in range(n2)] for i1 in range(n1)])
    Y[s1:e1+1, s2:e2+1] = Y_glob[s1:e1+1, s2:e2+1]
    Y.update_ghost_regions()

    # ...
    X_glob = kron_solve_seq_ref(A1, A2, Y_glob)

    comm.Barrier()
    wt = MPI.Wtime()
    X, lwt = kron_solve_bnd_par([A1_bnd, la1, ua1, A1_piv], [A2_bnd, la2, ua2, A2_piv], Y)
    wt = MPI.Wtime() - wt

    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank, ' - elapsed time: ', lwt, wt)
#            print('X_glob  = \n', X_glob)
#            print('X  = \n', X.toarray().reshape(n1, n2))
#            print('', flush=True)
#            time.sleep(0.1)
        #comm.Barrier()
    # ...

    # ... Check data
    #assert np.allclose( X[s1:e1+1, s2:e2+1], X_glob[s1:e1+1, s2:e2+1], rtol=1e-13, atol=1e-13 )
#===============================================================================

#===============================================================================
if __name__ == "__main__":

    n1 = 64
    n2 = 64
    p1 = 2
    p2 = 2

    test_kron_solver_2d_band_par(n1, n2, p1, p2)

