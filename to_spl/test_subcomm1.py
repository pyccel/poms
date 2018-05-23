# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace

# ...
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# ...

# ... 2D MPI cart
n1 = 8 ; n2 = 4
p1 = 1 ; p2 = 1
cart = Cart(npts = [n1, n2], pads = [p1, p2],  \
            periods = [False, False],\
            reorder = True, comm = comm)
# ...

V = StencilVectorSpace(cart)
s1, s2 = V.starts
e1, e2 = V.ends

Y = np.zeros((n1))
for i1 in range(s1, e1+1 ):
    Y[i1] = 10*(rank+1) + i1+1

subcomm_1 = V.cart._subcomm[0]


Y_glob = np.zeros((n1))

Y_loc = Y[s1:e1+1].copy()
subcomm_1.Iallgatherv(Y_loc, Y_glob)

for i in range(comm.Get_size()):
    if rank == i:
        print('---------------------')
        print('rank ', rank)
        print('Y  = ', Y)
        print('L  = ', Y_loc)
        print('G  = ', Y_glob)
        print('', flush=True)
    comm.Barrier()
# ...
