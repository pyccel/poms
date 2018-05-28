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


    for ie1 in range(se, ee+1) :
        i_span_1 = spans_1[ie1]

        for il_1 in range(0, p1+1):
            i1 = i_span_1 - p1  - 1 + il_1

            if s1 <= i1 <= e1:
                for jl_1 in range(0, p1+1):
                    j1   = i_span_1 - p1  - 1 + jl_1
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
    ne = 6
    p  = 2

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
    M = assembly_matrices(S)

    np.set_printoptions(linewidth=1000, precision=2)

   # ..
    for i in range(comm.Get_size()):
        if rank == i:
            print('rank= ', rank)
            print("A = \n", M.toarray())
#            print("A = \n", M._data)
            print('', flush=True)
        comm.Barrier()
    # ...
