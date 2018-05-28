# -*- coding: UTF-8 -*-
from numpy              import zeros, linspace
from spl.linalg.stencil import StencilMatrix
from spl.fem.splines    import SplineSpace
from spl.fem.tensor     import TensorFemSpace
from mpi4py             import MPI

# ... Assembly of the elementary matrix
# ... the weak form of: - (uxx + uyy) + u
def kernel(p1, p2, k1, k2, bs1, bs2, w1, w2, mat):
    mat[:,:,:,:] = 0.
    for il_1 in range(0, p1+1):
        for jl_1 in range(0, p1+1):
            for il_2 in range(0, p2+1):
                for jl_2 in range(0, p2+1):

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = bs1[il_1, 0, g1] * bs2[il_2, 0, g2]
                            bi_x = bs1[il_1, 1, g1] * bs2[il_2, 0, g2]
                            bi_y = bs1[il_1, 0, g1] * bs2[il_2, 1, g2]

                            bj_0 = bs1[jl_1, 0, g1] * bs2[jl_2, 0, g2]
                            bj_x = bs1[jl_1, 1, g1] * bs2[jl_2, 0, g2]
                            bj_y = bs1[jl_1, 0, g1] * bs2[jl_2, 1, g2]

                            wvol = w1[g1] * w2[g2]

                            v += (bi_0*bj_0 + bi_x*bj_x + bi_y*bj_y)*wvol
                    mat[il_1, il_2, p1 + jl_1 - il_1, p2 + jl_2 - il_2] = v
# ...

# ... Assembly of the stifness matrix
def assembly(V, kernel):

    # ... sizes
    [s1, s2] = V.vector_space.starts
    [e1, e2] = V.vector_space.ends
    [p1, p2] = V.vector_space.pads
    # ...

    # ... seetings
    [k1, k2] = [W.quad_order for W in V.spaces]
    [spans_1, spans_2] = [W.spans for W in V.spaces]
    [basis_1, basis_2] = [W.basis for W in V.spaces]
    [weights_1, weights_2] = [W.weights for W in V.spaces]
    [points_1, points_2] = [W.points for W in V.spaces]
    # ...

    # ... data structure
    M = StencilMatrix(V.vector_space, V.vector_space)
    # ...

    # ... element matrix
    #mat = zeros((p1+1, p2+1, 2*p1+1, 2*p2+1), order='F')
    mat = zeros((e1-s1+1, e2-s2+1, 2*p1+1, 2*p2+1), order='F')
    # ...

    # ...
    current_rank = V.vector_space.cart._rank
    last_rank    = V.vector_space.cart._size - 1

    if current_rank == last_rank == 0:
        se1 = s1
        ee1 = e1 - p1
        se2 = s2
        ee2 = e2 - p2
    elif current_rank == 0:
        se1 = s1
        ee1 = e1
        se2 = s2
        ee2 = e2
    elif current_rank == last_rank:
        se1 = s1 - p1
        ee1 = e1 - p1
        se2 = s2 - p2
        ee2 = e2 - p2
    else:
        se1 = s1 - p1
        ee1 = e1
        se2 = s2 - p2
        ee2 = e2
    # ...

    # ... build matrices
    for ie1 in range(se1, ee1+1):
        i_span_1 = spans_1[ie1]

        for ie2 in range(se2, ee2+1):
            i_span_2 = spans_2[ie2]

            i1 = i_span_1 - p1 - 1
            i2 = i_span_2 - p2 - 1

            if (s1 <= i1 <= e1) and (s2 <= i2 <= e2):
                bs1 = basis_1[:, :, :, ie1]
                bs2 = basis_2[:, :, :, ie2]
                w1 = weights_1[:, ie1]
                w2 = weights_2[:, ie2]

                for il_1 in range(0, p1+1):
                    for jl_1 in range(0, p1+1):
                        for il_2 in range(0, p2+1):
                            for jl_2 in range(0, p2+1):

                                # ...
                                v = 0.0
                                for g1 in range(0, k1):
                                    for g2 in range(0, k2):
                                        bi_0 = bs1[il_1, 0, g1] * bs2[il_2, 0, g2]
                                        bi_x = bs1[il_1, 1, g1] * bs2[il_2, 0, g2]
                                        bi_y = bs1[il_1, 0, g1] * bs2[il_2, 1, g2]

                                        bj_0 = bs1[jl_1, 0, g1] * bs2[jl_2, 0, g2]
                                        bj_x = bs1[jl_1, 1, g1] * bs2[jl_2, 0, g2]
                                        bj_y = bs1[jl_1, 0, g1] * bs2[jl_2, 1, g2]

                                        wvol = w1[g1] * w2[g2]

                                        v += (bi_0*bj_0 + bi_x*bj_x + bi_y*bj_y)*wvol
                                # ...
                                M[i1, i2, j1 - i1, j2 - i2]  += v
#                kernel(p1, p2, k1, k2, bs1, bs2, w1, w2, mat)
#                M[i1:i1+e1-s1+1,i2:i2+e2-s2+1,:,:] += mat[:,:,:,:]
    # ...

    return M
# ...

