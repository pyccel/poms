# -*- coding: UTF-8 -*-
from mpi4py             import MPI
import numpy as np
from numpy              import zeros, ones, linspace
from numpy.linalg       import norm
from spl.linalg.stencil import StencilMatrix, StencilVector
from spl.fem.splines    import SplineSpace
from spl.fem.tensor     import TensorFemSpace

from spl.linalg.solvers import cg
from matrix_assembler_2d   import kernel, assembly


###############################################################################
if __name__ == '__main__':


    # ... Fine Grid: numbers of elements and degres
    p1  = 1 ; p2  = 1
    ne1 = 4 ; ne2 = 4
    # ...

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print('> Grid   :: [{ne1},{ne2}]'.format(ne1=ne1, ne2=ne2))
        print('> Degree :: [{p1},{p2}]'.format(p1=p1, p2=p2))

    grid_1 = linspace(0., 1., ne1+1)
    grid_2 = linspace(0., 1., ne2+1)

    S1 = SplineSpace(p1, grid=grid_1)
    S2 = SplineSpace(p2, grid=grid_2)

    S = TensorFemSpace(S1, S2, comm=comm)
    V = S.vector_space

    s1, s2 = V.starts
    e1, e2 = V.ends


    # ... Weak form: compute the stifeness matrix
    A = assembly(S)

    x = StencilVector(V)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1, e1+1):
        for i2 in range(s2, e2+1):
            x[i1,i2] = i1+i2
    x.update_ghost_regions()

    #... Compute matrix-vector product
    b = A.dot(x)

    np.set_printoptions(linewidth=10000, precision=1)

   # ..
    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print("A = \n", A.toarray())
#            print("x = \n", x.toarray())
#            print("b = \n", b.toarray())
            print('', flush=True)
        comm.Barrier()
    # ...
