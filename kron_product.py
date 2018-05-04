# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix

# ... TODO use the SPL version when available
def update_ghost_regions(stencil_vector, direction):

    # ...
    data  = stencil_vector._data
    space = stencil_vector._space
    cart  = space.cart
    comm  = cart.comm_cart

    # ... Choose non-negative invertible function tag(disp) >= 0
    # ... NOTE: different values of disp must return different tags!
    tag = lambda disp: 42+disp

    # Requests' handles
    requests = []

    # ... Start receiving data (MPI_IRECV)
    for disp in [-1,1]:
        info     = cart.get_shift_info(direction, disp)
        recv_typ = space.get_recv_type(direction, disp)
        recv_buf = (data, 1, recv_typ)
        recv_req = comm.Irecv(recv_buf, info['rank_source'], tag(disp))
        requests.append(recv_req)

    # ... Start sending data (MPI_ISEND)
    for disp in [-1,1]:
        info     = cart.get_shift_info(direction, disp)
        send_typ = space.get_send_type(direction, disp)
        send_buf = (data, 1, send_typ)
        send_req = comm.Isend(send_buf, info['rank_dest'], tag(disp))
        requests.append(send_req)

    # ... Wait for end of data exchange (MPI_WAITALL)
    MPI.Request.Waitall(requests)
# ...

# .. Fill in stencil matrix as band
def populate_1d_matrix(M):
    e = M.ends[0]
    s = M.starts[0]
    p = M.pads[0]

    for i in range(s, e+1):
        for k in range(-p, p+1):
            M[i, k] = k
        M[i, 0] = 5.*p
    M.remove_spurious_entries()
# ...

n1 = 16
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
populate_1d_matrix(A)
populate_1d_matrix(B)


# ... Exchange Ghosts cell for X
X[:, :] = 1.
X.update_ghost_regions()

# ... Init (populate X, A and B)
Y[:,:]    = 0.

#print('rank: ',rank, 'X: ', X._data.shape,  'Xt: ', Xtmp._data.shape, 'A: ', A._data.shape[0], 'B: ', B._data.shape[0])
# ... Compute Kron prod
for i1 in range(V1.npts[0]):

    for i2 in range(V2.npts[0]):

        Xtmp[i1, i2] = 0.

        for k2 in range(-p2, p2+1):
            j2 = i2 + k2
            Xtmp[i1, i2] += X[i1, j2]*A[i2, k2]

        update_ghost_regions(Xtmp, direction=1)
        update_ghost_regions(Xtmp, direction=0)

        # ...
        for k1 in range(-p1, p1+1):
            j1 = i1 + k1
            Y[i1, i2] += B[i1, k1]*Xtmp[j1, i2]

        update_ghost_regions(Y, direction=0)
        update_ghost_regions(Y, direction=1)
        # ...
# ...

for i in range(comm.Get_size()):
    if rank == i:
        print('rank= ', rank)
        print('A=', A.toarray())
        print('B=', B.toarray())
        print('X=', X.toarray())
        print('Y=', Y.toarray())
        print('', flush=True)
    comm.Barrier()


#print('>> ',rank, cart.coords)
#print('pid: ',rank, 'X: ', X._data.shape, 'A: ', A._data.shape[0], 'B: ', B._data.shape[0])


