# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix


# ...
n = 8
m = 8
p = 1
q = 1
# ...

# ..
comm = MPI.COMM_WORLD

cart1 = Cart(npts    = [n,],
             pads    = [p,],
             periods = [True,],
             reorder = False,
             comm    = comm )

cart2 = Cart(npts    = [m,],
             pads    = [q,],
             periods = [True,],
             reorder = False,
             comm    = comm )

cart3 = Cart(npts    = [m,n],
             pads    = [q,p],
             periods = [True, True],
             reorder = False,
             comm    = comm )
# ..

# ..
V1 = StencilVectorSpace(cart1)
V2 = StencilVectorSpace(cart2)
W  = StencilVectorSpace(cart3)

A = StencilMatrix(V1, V1)
B = StencilMatrix(V2, V2)
X = StencilVector(W)
Y = StencilVector(W)
# ..


