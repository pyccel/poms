# -*- coding: UTF-8 -*-

import numpy as np
from numpy.linalg       import norm
from mpi4py             import MPI
from spl.linalg.stencil import StencilMatrix, StencilVector
from spl.fem.splines    import SplineSpace
from spl.fem.tensor     import TensorFemSpace

from solvers            import pcg, jacobi, damped_jacobi
from matrix_assembler   import assembly

'''
Test of parallel cg of Ax=b, A is Laplacian matrix
'''


###############################################################################
if __name__ == '__main__':


    # ... Fine Grid: numbers of elements and degres
    p1  = 1 ; p2  = 1
    ne1 = 16 ; ne2 = 16
    # ...

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print('> Grid   :: [{ne1},{ne2}]'.format(ne1=ne1, ne2=ne2))
        print('> Degree :: [{p1},{p2}]'.format(p1=p1, p2=p2))

    grid_1 = np.linspace(0., 1., ne1+1)
    grid_2 = np.linspace(0., 1., ne2+1)

    S1 = SplineSpace(p1, grid=grid_1)
    S2 = SplineSpace(p2, grid=grid_2)

    S = TensorFemSpace(S1, S2, comm=comm)
    V = S.vector_space

    s1, s2 = V.starts
    e1, e2 = V.ends


    # ... Weak form: compute the stifeness matrix
    A = assembly(S)
    A = 1./(ne1+ne2)*A
    x0 = StencilVector(V)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1, e1+1):
        for i2 in range(s2, e2+1):
            x0[i1,i2] = i1 + i2 + 1.
    x0.update_ghost_regions()

    #... Compute matrix-vector product
    b = A.dot(x0)

    # ... Solve the system
    wt = MPI.Wtime()

    pc = eval('damped_jacobi')
    x1, info = pcg(A, pc, b, tol=1e-8, maxiter=1000, verbose=False)
    wt = MPI.Wtime() - wt

    # ...
    err1 = x1-x0

    np.set_printoptions(linewidth=10000, precision=3)
   # ..
    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print("x0 = \n", x0.toarray())
#            print("b = \n", b.toarray())
            print("x1 = \n", x1.toarray())
#            print("A = \n", A.toarray())
            print("info= ", info)
            print("err_norm= ", norm(err1.toarray()))
            print('elapsed time: {}'.format(wt))
            print('', flush=True)
        comm.Barrier()
    # ...
