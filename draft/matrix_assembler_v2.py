# -*- coding: UTF-8 -*-
import numpy as np
from spl.linalg.stencil import StencilMatrix
from spl.fem.splines    import SplineSpace
from spl.fem.tensor     import TensorFemSpace
from mpi4py             import MPI


def kernel(r, ks1, ke1, p1, ks2, ke2, p2, k1, k2, bs1, bs2, w1, w2, mat):
    for il_1 in range(ks1, ke1+1):
        for jl_1 in range(0, p1+1):
            for il_2 in range(ks2, ke2+1):
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
                    if r==1:
                        print(il_1, il_2,ks2, ke2)
                    mat[il_1, il_2, p1 + jl_1 - il_1, p2 + jl_2 - il_2] = v
# ...

# ...
def local_element_idx(start, end, pad, coord, n_proc):

    if coord == 0:
        se = start
        ee = end
    elif coord == n_proc - 1:
        se = start - pad
        ee = end   - pad
    else:
        se = start - pad
        ee = end

    return se, ee
# ...

# ... Assembly of the stifness matrix
# ... the weak form of: - (uxx + uyy) + u
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

    # ... distributed ends elements
    rank   = V.vector_space.cart._rank
    procs  = V.vector_space.cart.nprocs
    coords = V.vector_space.cart.coords

    # ... monoproc case
    if procs == [1, 1]:
        se1 = s1
        ee1 = e1 - p1
        se2 = s2
        ee2 = e2 - p2
    else:
        se1, ee1 =  local_element_idx(s1, e1, p1, coords[0], procs[0])
        se2, ee2 =  local_element_idx(s2, e2, p2, coords[1], procs[1])
    # ...

    print(rank, [(s1, e1), (se1, ee1)], [(s2, e2), (se2, ee2)])

    # ... build matrices
    for ie1 in range(se1, ee1+1):
        i_span_1 = spans_1[ie1]
        ks1 = max(0 , s1 - i_span_1 + p1 + 1)
        ke1 = min(p1, e1 - i_span_1 + p1 + 1)

        for ie2 in range(se2, ee2+1):
            i_span_2 = spans_2[ie2]
            ks2 = max(0 , s2 - i_span_2 + p2 + 1)
            ke2 = min(p2, e2 - i_span_2 + p2 + 1)
            if rank==1:
                print(ie2, i_span_2)
            bs1 = basis_1[:, :, :, ie1]
            bs2 = basis_2[:, :, :, ie2]
            w1 = weights_1[:, ie1]
            w2 = weights_2[:, ie2]

            mat = np.zeros((ke1+1-ks1, ke2+1-ks2, 2*p1+1, 2*p2+1), order='F')
            kernel(rank, ks1, ke1, p1, ks2, ke2, p2, k1, k2, bs1, bs2, w1, w2, mat)

            i1 = i_span_1 - p1 - 1
            i2 = i_span_2 - p2 - 1

            M[i1+ks1:i1+ke1+1, i2+ks2:i2+ke2+1,:,:] += mat[:,:,:,:]
            # ...


    # ...

    return M
# ...

