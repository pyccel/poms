# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix

# ...
n1 = 16
n2 = 32
p1 = 1
p2 = 1
# ...

# ..
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# ...

# ... 2D MPI cart
cart = Cart(npts=[n1, n2], pads=[p1, p2], periods=[True, True], reorder=True, comm=comm)

# ...
V = StencilVectorSpace(cart)
X = StencilVector(V)
Y = StencilVector(V)
# ...

# ...
V1 = StencilVectorSpace([cart.ends[0] - cart.starts[0] + 1], [p1])
V2 = StencilVectorSpace([cart.ends[1] - cart.starts[1] + 1], [p2])

A = StencilMatrix(V1, V1)
B = StencilMatrix(V2, V2)
# ..
#print('>> ',rank, cart.coords)
print('pid: ',rank, 'X: ', X._data.shape, 'A: ', A._data.shape[0], 'B: ', B._data.shape[0])
