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
def populate_2d_vector(X):
    e1 = X.ends[0]
    s1 = X.starts[0]
    e2 = X.ends[1]
    s2 = X.starts[1]

    for i1 in range(s1, e1+1 ):
        for i2 in range(s2, e2+1):
            X[i1,i2] = i1 + i2
# ...

# ... Compute Y = (A kron B) X
def kron_dot(A, B,  X):
    # ...
    V = X.space
    Xtmp = StencilVector(V)

    Y    = StencilVector(V)
    Y[:,:] = 0.
    # ...

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

    return Y

# ...

# ...
def kron_dot_scipy(A, B, X):
    from scipy import sparse
    import numpy as np

    # ...
    A_csr = A.tocsr()
    B_csr = B.tocsr()
    C = sparse.kron(A_csr, B_csr)
    # ...

    # ...
    e1 = X.ends[0]
    s1 = X.starts[0]
    e2 = X.ends[1]
    s2 = X.starts[1]
    X_arr = np.zeros((e1+1-s1)*(e2+1-s2))
    i = 0
    for i1 in range(s1, e1+1):
        for i2 in range(s2, e2+1):
            X_arr[i] = X[i1,i2]
            i += 1
    # ...

    Y_arr = C.dot(X_arr)

    return Y_arr
# ...

# ...
#def kron_solve(A, B, X):
# ipython save
# ...


#===============================================================================
if __name__ == "__main__":
    # ...
    n1 = 4
    n2 = 4
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

    #populate_2d_vector(X)

    X[:, :] = 1.

    Y = kron_dot(A, B, X)

    Ys = kron_dot_scipy(A, B, X)

    # ...
    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print('A=', A.toarray())
            print('B=', B.toarray())
            print('X=', X.toarray())
            print('Y=', Y.toarray())
            print('Ys=', Ys)
            print('', flush=True)
        comm.Barrier()
    # ...


