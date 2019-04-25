# -*- coding: UTF-8 -*-
from numpy              import zeros, linspace
from psydac.linalg.stencil import StencilMatrix,StencilVector
from psydac.fem.splines    import SplineSpace
from psydac.fem.tensor     import TensorFemSpace
from mpi4py             import MPI


def kernel( p, k, bs, w, mat1, mat2):

    mat1[:,:] = 0.
    mat2[:,:] = 0.

    for i in range(p+1):
        for j in range(p+1):

            bi_1 = bs[i, 1, :]
            bj_1 = bs[j, 1, :]
            bi   = bs[i, 0, :]
            bj   = bs[j, 0, :]
            wvol = w[:]


            v_m  = bi * bj * wvol
            v_s  = bi_1*bj_1*wvol

            mat1[i, p+j-i] = v_m.sum()
            mat2[i, p+j-i] = v_s.sum()

def assembly_1d(V):

    [p1] = V.degree
    
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
   
    # Quadrature data
    ne        = V.quad_grids[0].num_elements
    k1        = V.quad_grids[0].num_quad_pts
    spans_1   = V.quad_grids[0].spans
    basis_1   = V.quad_grids[0].basis
    weights_1 = V.quad_grids[0].weights

    mass = StencilMatrix( V.vector_space, V.vector_space )
    stif = StencilMatrix( V.vector_space, V.vector_space )

    # Create element matrices
    mat_m1 = zeros((p1+1,2*p1+1))
    mat_s1 = zeros((p1+1,2*p1+1))

    
    # Build global matrices: cycle over elements

    for ie1 in range(ne):

        # Get spline index, B-splines' values and quadrature weights
        is1 = spans_1[ie1]
        bs1 = basis_1[ie1,:,:,:]
        w   = weights_1[ie1,:]

        kernel(p1, k1, bs1, w, mat_m1, mat_s1)
        mass[is1-p1:is1+1,:] += mat_m1[:,:]
        stif[is1-p1:is1+1,:] += mat_s1[:,:]

        
    mass.remove_spurious_entries()
    stif.remove_spurious_entries()
    
    return stif, mass

def assemble_rhs( V, f ):
    """
    Assemble right-hand-side vector.

    Parameters
    ----------
    V : SplineSpace
        Finite element space where the Galerkin method is applied.

    f : callable
        Right-hand side function (charge density).

    Returns
    -------
    rhs : StencilVector
        Vector b of coefficients, in linear system Ax=b.

    """
    # Sizes
    [s1] = V.vector_space.starts
    [e1] = V.vector_space.ends
    [p1] = V.vector_space.pads

    # Quadrature data
    nk1       = V.quad_grids[0].num_elements
    nq1       = V.quad_grids[0].num_quad_pts
    spans_1   = V.quad_grids[0].spans
    basis_1   = V.quad_grids[0].basis
    points_1  = V.quad_grids[0].points
    weights_1 = V.quad_grids[0].weights

    # Data structure
    rhs = StencilVector( V.vector_space )

    # Build RHS
    for k1 in range( nk1 ):

        is1    =   spans_1[k1]
        bs1    =   basis_1[k1,:,:,:]
        x1     =  points_1[k1, :]
        wvol   = weights_1[k1, :]
        f_quad = f( x1 )

        for il1 in range( p1+1 ):

            bi_0 = bs1[il1, 0, :]
            v    = bi_0 * f_quad * wvol

            rhs[is1-p1+il1] += v.sum()

    return rhs


# ... Assembly of the stifness matrix
# ... the weak form of: - (uxx + uyy) + u
def assembly_2d(V):

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

    # ...
    def local_element_idx(start, end, pad, coord, n_proc):
        if coord == 0:
            se = start
            ee = end
        elif coord == n_proc-1:
            se = start - pad
            ee = end - pad
        else:
            se = start - pad
            ee = end

        return se, ee
    # ...

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

    # ... build matrices
    for ie1 in range(se1, ee1+1):
        i_span_1 = spans_1[ie1]
        ks1 = max(0 , s1 - i_span_1 + p1 + 1)
        ke1 = min(p1, e1 - i_span_1 + p1 + 1)

        for ie2 in range(se2, ee2+1):
            i_span_2 = spans_2[ie2]
            ks2 = max(0 , s2 - i_span_2 + p2 + 1)
            ke2 = min(p2, e2 - i_span_2 + p2 + 1)

            for il_1 in range(ks1, ke1+1):
                i1 = i_span_1 - p1  - 1 + il_1

                for jl_1 in range(0, p1+1):
                    j1 = i_span_1 - p1  - 1 + jl_1

                    for il_2 in range(ks2, ke2+1):
                        i2 = i_span_2 - p2 - 1 + il_2

                        for jl_2 in range(0, p2+1):
                            j2 = i_span_2 - p2  - 1 + jl_2

                            # ...
                            v = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):

                                    bi_0 = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                                    bi_x = basis_1[il_1, 1, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                                    bi_y = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 1, g2, ie2]

                                    bj_0 = basis_1[jl_1, 0, g1, ie1] * basis_2[jl_2, 0, g2, ie2]
                                    bj_x = basis_1[jl_1, 1, g1, ie1] * basis_2[jl_2, 0, g2, ie2]
                                    bj_y = basis_1[jl_1, 0, g1, ie1] * basis_2[jl_2, 1, g2, ie2]

                                    wvol = weights_1[g1, ie1] * weights_2[g2, ie2]

                                    v += (bi_0*bj_0 + bi_x*bj_x + bi_y*bj_y)*wvol
                            # ...

                            M[i1, i2, j1 - i1, j2 - i2]  += v
    # ...

    return M
# ...

# ....
def assembly_2d_seq(V):

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

    # ... monoproc case
    se1 = s1
    ee1 = e1 - p1
    se2 = s2
    ee2 = e2 - p2

    # ... build matrices
    for ie1 in range(se1, ee1+1):
        i_span_1 = spans_1[ie1]
        ks1 = max(0 , s1 - i_span_1 + p1 + 1)
        ke1 = min(p1, e1 - i_span_1 + p1 + 1)

        for ie2 in range(se2, ee2+1):
            i_span_2 = spans_2[ie2]
            ks2 = max(0 , s2 - i_span_2 + p2 + 1)
            ke2 = min(p2, e2 - i_span_2 + p2 + 1)

            for il_1 in range(ks1, ke1+1):
                i1 = i_span_1 - p1  - 1 + il_1

                for jl_1 in range(0, p1+1):
                    j1 = i_span_1 - p1  - 1 + jl_1

                    for il_2 in range(ks2, ke2+1):
                        i2 = i_span_2 - p2 - 1 + il_2

                        for jl_2 in range(0, p2+1):
                            j2 = i_span_2 - p2  - 1 + jl_2

                            # ...
                            v = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):

                                    bi_0 = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                                    bi_x = basis_1[il_1, 1, g1, ie1] * basis_2[il_2, 0, g2, ie2]
                                    bi_y = basis_1[il_1, 0, g1, ie1] * basis_2[il_2, 1, g2, ie2]

                                    bj_0 = basis_1[jl_1, 0, g1, ie1] * basis_2[jl_2, 0, g2, ie2]
                                    bj_x = basis_1[jl_1, 1, g1, ie1] * basis_2[jl_2, 0, g2, ie2]
                                    bj_y = basis_1[jl_1, 0, g1, ie1] * basis_2[jl_2, 1, g2, ie2]

                                    wvol = weights_1[g1, ie1] * weights_2[g2, ie2]

                                    v += (bi_0*bj_0 + bi_x*bj_x + bi_y*bj_y)*wvol
                            # ...

                            M[i1, i2, j1 - i1, j2 - i2]  += v
    # ...

    return M
# ...

