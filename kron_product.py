# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix

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

# ...
n1 = 8
n2 = 8
p1 = 2
p2 = 1
# ...

# ..
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# ...

# ... 2D MPI cart
cart = Cart(npts=[n1, n2], pads=[p1, p2], periods=[False, False], reorder=True, comm=comm)

# ...
V    = StencilVectorSpace(cart)
X    = StencilVector(V)
Xtmp = StencilVector(V)
Y    = StencilVector(V)
# ...

# ...
V1 = StencilVectorSpace([cart.ends[0] - cart.starts[0] + 1], [cart.pads[0]], [cart.periods[0]])
V2 = StencilVectorSpace([cart.ends[1] - cart.starts[1] + 1], [cart.pads[1]], [cart.periods[1]])

B = StencilMatrix(V1, V1)
A = StencilMatrix(V2, V2)
# ..

# ... Populate A, B and X
populate_1d_matrix(A, 5.)
populate_1d_matrix(B, 8.)


# ... Exchange Ghosts cell for X
X[:, :] = 1.
X.update_ghost_regions()

# ... Init (populate X, A and B)
Y[:,:]    = 0.

#print('rank: ',rank, 'X: ', X._data.shape,  'Xt: ', Xtmp._data.shape, 'A: ', A._data.shape[0], 'B: ', B._data.shape[0])
# ... Compute Kron prod
for i1 in range(V1.npts[0]):

    i1_glob = i1 + V.starts[0]

    for i2 in range(V2.npts[0]):
        i2_glob = i2 + V.starts[1]

        Xtmp[i1_glob, i2_glob] = 0.

        for k2 in range(-p2, p2+1):
            j2_glob = i2_glob + k2
            Xtmp[i1_glob, i2_glob] += X[i1_glob, j2_glob]*A[i2, k2]

        Xtmp.update_ghost_regions(direction=1)

        # ...
        for k1 in range(-p1, p1+1):
            j1_glob = i1_glob + k1
            Y[i1_glob, i2_glob] += B[i1, k1]*Xtmp[j1_glob, i2_glob]

        Y.update_ghost_regions(direction=0)
        # ...
# ...

for i in range(comm.Get_size()):
    if rank == i:
#        print('rank= ', rank)
#        print('A=', A.toarray())
#        print('B=', B.toarray())
#        print('X=', X.toarray())
#        print('Y=', Y.toarray())
#        print('================')
        print('n1 = ', n1)
        print('nv = ', cart.ends[0] - cart.starts[0] + 1)
        print('', flush=True)
    comm.Barrier()


#print('>> ',rank, cart.coords)
#print('pid: ',rank, 'X: ', X._data.shape, 'A: ', A._data.shape[0], 'B: ', B._data.shape[0])


