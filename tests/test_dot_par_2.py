# -*- coding: UTF-8 -*-
from mpi4py             import MPI
import numpy as np
from numpy              import zeros, ones, linspace
from numpy.linalg       import norm
from spl.linalg.stencil import StencilMatrix, StencilVector, StencilVectorSpace
from spl.fem.splines    import SplineSpace
from spl.fem.tensor     import TensorFemSpace
from spl.ddm.cart import Cart

from spl.linalg.solvers import cg
from matrix_assembler   import kernel, assembly


###############################################################################
if __name__ == '__main__':


    # ... Fine Grid: numbers of elements and degres
    p1  = 1 ; p2  = 1
    ne1 = 3 ; ne2 = 3
    n1 = ne1+p1
    n2 = ne2+p2
    # ...

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cart = Cart( npts    = [n1, n2],
                 pads    = [p1, p2],
                 periods = [False, False],
                 reorder = True,
                 comm    = comm )

    # Create vector space, stencil matrix, and stencil vector
    V = StencilVectorSpace( cart )
    M = StencilMatrix( V, V )
    x = StencilVector( V )

    s1,s2 = V.starts
    e1,e2 = V.ends

    # Fill in stencil matrix values based on diagonal index (periodic!)
    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M[:,:,k1,k2] = 10*k1+k2

    # If any dimension is not periodic, set corresponding periodic corners to zero
    M.remove_spurious_entries()

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1,e1+1):
        for i2 in range(s2,e2+1):
            x[i1,i2] = 1.0
    x.update_ghost_regions()

    # Compute matrix-vector product
    y = M.dot(x)
    np.set_printoptions(linewidth=10000, precision=3)

   # ..
    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print("A = \n", M.toarray())
            print("x = \n", x.toarray())
            print("b = \n", y.toarray())
            print('', flush=True)
        comm.Barrier()
    # ...
