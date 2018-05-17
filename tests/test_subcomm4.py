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
n1 = 8 ; n2 = 8
p1 = 1 ; p2 = 1
cart = Cart(npts = [n1, n2], pads = [p1, p2],  \
            periods = [False, False],\
            reorder = True, comm = comm)
# ...

V = StencilVectorSpace(cart)
s1, s2 = V.starts
e1, e2 = V.ends

X = np.zeros((n1))
for i1 in range(s1, e1+1 ):
    X[i1] = 10*(rank+1) + i1+1

Y = np.zeros((n2))
for i2 in range(s2, e2+1 ):
    Y[i2] = 10*(rank+1) + i2+1

subcomm_1 = V.cart._subcomm[0]
subcomm_2 = V.cart._subcomm[1]

X_glob = np.zeros((n1))
X_loc = X[s1:e1+1].copy()
subcomm_1.Allgatherv(X_loc, X_glob)

Y_glob = np.zeros((n2))
Y_loc = X_glob[s2:e2+1].copy()
subcomm_2.Allgatherv(Y_loc, Y_glob)

for i in range(comm.Get_size()):
    if rank == i:
        print('---------------------')
        print('rank ', rank)
        print('X_LOC  = ', X_loc)
        print('X_GLOB = ', X_glob)
        print('Y_LOC  = ', Y_loc)
        print('Y_GLOB = ', Y_glob)
        print('', flush=True)
    comm.Barrier()
# ...
