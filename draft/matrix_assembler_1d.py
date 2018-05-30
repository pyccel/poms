# -*- coding: UTF-8 -*-
import numpy   as np
from spl.linalg.stencil import StencilMatrix
from spl.linalg.stencil import StencilVectorSpace
from spl.fem.splines    import SplineSpace
from mpi4py             import MPI
from spl.ddm.cart       import Cart

# ...
def assembly_matrices(V):

    # ... sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    k1 = V.quad_order
    spans_1 = V.spans
    basis_1 = V.basis
    weights_1 = V.weights
    # ...

    # ... data structure
    mass      = StencilMatrix( V.vector_space, V.vector_space )
    stiffness = StencilMatrix( V.vector_space, V.vector_space )
    # ...

    # ... build matrices
    current_rank = V.vector_space.cart._rank
    last_rank    = V.vector_space.cart._size-1


    if current_rank == last_rank == 0:
        se = s1
        ee = e1 - p1
    elif current_rank == 0:
        se = s1
        ee = e1
    elif current_rank == last_rank:
        se = s1 - p1
        ee = e1 - p1
    else:
        se = s1 - p1
        ee = e1

    print(current_rank, (s1, e1),V.vector_space.cart._size)

    for ie1 in range(se, ee+1) :
        is1 = spans_1[ie1]
        k  = is1 - p1 - 1
        ks = max(0, s1-k)
        ke = min(p1, e1-k)

        for il_1 in range(ks, ke+1):
            i1 = is1 - p1  - 1 + il_1

            for jl_1 in range(0, p1+1):
                j1   = is1 - p1  - 1 + jl_1
                v_m = 0.0
                v_s = 0.0

                for g1 in range(0, k1):
                    bi_0 = basis_1[il_1, 0, g1, ie1]
                    bi_x = basis_1[il_1, 1, g1, ie1]

                    bj_0 = basis_1[jl_1, 0, g1, ie1]
                    bj_x = basis_1[jl_1, 1, g1, ie1]

                    wvol = weights_1[g1, ie1]

                    v_m += bi_0 * bj_0 * wvol
                    v_s += (bi_x * bj_x) * wvol

                mass[i1, j1 - i1] += v_m
                stiffness[i1, j1 - i1]  += v_s

    # ...

    return stiffness

# ...

###############################################################################
if __name__ == '__main__':

    # ...
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # ...
    ne = 4
    p  = 1

    grid = np.linspace(0., 1., ne+1)
    S = SplineSpace(p, grid=grid)
    n = S.nbasis

    if rank == 0:
        print('> Grid   :: {ne}'.format(ne=ne))
        print('> Degree :: {p}'.format(p=p))
        print('> nbasis :: {n}'.format(n=n))

    # ... Vector Spaces
    cart = Cart(npts =[n], pads = [p], periods = [False],\
                reorder = True, comm = comm)

    S._vector_space = StencilVectorSpace(cart)
    S._initialize()

    # ... Stifness Matrix
    wt = MPI.Wtime()
    M = assembly_matrices(S)
    wt = MPI.Wtime() - wt

    np.set_printoptions(linewidth=1000, precision=2)

   # ..
    for i in range(comm.Get_size()):
        if rank == i:
            print('Rank= ', rank)
            print(M.toarray())
            print('Elapsed time= {}'.format(wt))
            print('', flush=True)
        comm.Barrier()
    # ...
