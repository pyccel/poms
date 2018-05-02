# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix

# ...
n = 8
m = 10
p = 1
q = 1
# ...

# ..
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# ...

# ... 2D MPI cart
cart = Cart(npts    = [m,n],
            pads    = [q,p],
            periods = [True, True],
            reorder = False,
            comm    = comm)

# ... 1D MPI cart: rows or cols ?
comm1 = cart._subcomm[0]
cart1 = Cart(npts    = [n,],
             pads    = [p,],
             periods = [True,],
             reorder = False,
             comm    = comm1)
# ...

# ... 1D MPI cart: rows or cols ?
comm2 = cart._subcomm[1]
cart2 = Cart(npts    = [m,],
             pads    = [q,],
             periods = [True,],
             reorder = False,
             comm    = comm2)
# ..

# ..
V1 = StencilVectorSpace(cart1)
V2 = StencilVectorSpace(cart2)
W  = StencilVectorSpace(cart)

A = StencilMatrix(V1, V1)
B = StencilMatrix(V2, V2)
X = StencilVector(W)
Y = StencilVector(W)
# ..

if rank == 0:
    A[:,-1] = -1.0
    A[:, 0] =  5.0
    A[:,+1] =  1.0

    B[:,-1] = -2.0
    B[:, 0] =  4.0
    B[:,+1] =  2.0

    print('>> A: ', A.toarray())
    print('>> B: ', B.toarray())
