# -*- coding: UTF-8 -*-
import sys
import numpy as np
from mpi4py              import MPI
from scipy.sparse        import kron, csc_matrix, csr_matrix
from scipy.sparse.linalg import splu

from psydac.linalg.stencil  import StencilMatrix, StencilVector, StencilVectorSpace
from psydac.linalg.kron     import KroneckerStencilMatrix, KroneckerStencilMatrix_2D
from psydac.fem.splines     import SplineSpace
from psydac.fem.tensor      import TensorFemSpace
from psydac.core.interface  import make_open_knots, matrix_multi_stages, collocation_cardinal_splines
from matrix_assembler    import assembly_2d_seq
from multilevels         import knots_to_insert
from solvers             import pcg, damped_jacobi, pcg_glt, stencil_mat_dot_csc, csr_dot_csc, csr_dot_stencil_vec_2d
from utils               import  array_to_vect_stencil, array_to_mat_stencil
from matrix_assembler    import assembly_1d
from psydac.linalg.iterative_solvers import weighted_jacobi, cg, pcg

np.set_printoptions(linewidth=1000, precision=4)

# ...
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ... Spline degree and number of elements
p  = [2, 2]
n  = [3, 3]
nf = [2**n[0] +p[0], 2**n[1] +p[1]]
ndim = 2

# ... Knot vectors
Tf = [make_open_knots(p[0], nf[0]), make_open_knots(p[0], nf[0])]

V1 ,V2 = [SplineSpace(p[0], knots=Tf[0]), SplineSpace(p[1], knots=Tf[1])]

Sf1d  = [TensorFemSpace(V1), TensorFemSpace(V2)]
Sf    = TensorFemSpace(V1, V2, comm=comm)

Af1d1, mass1 = assembly_1d(Sf1d[0])
Af1d2, mass2 = assembly_1d(Sf1d[1])

Af1d = [Af1d1, Af1d2]


if not V1.periodic:
    Af1d[0][0,:] = 0
    Af1d[0][0,0] = 1.

    Af1d[0][V1.nbasis-1,:] = 0
    Af1d[0][V1.nbasis-1,0] = 1.
    
if not V2.periodic:
    Af1d[1][0,:] = 0
    Af1d[1][0,0] = 1.

    Af1d[1][V1.nbasis-1,:] = 0
    Af1d[1][V1.nbasis-1,0] = 1.


Af = KroneckerStencilMatrix_2D(Sf.vector_space, Sf.vector_space, *Af1d)

T      = [Tf]
P      = []
R      = []
A      = [Af]
V      = [Sf.vector_space]
starts = [V[0].starts]
ends   = [V[0].ends]

coords = Sf._vector_space.cart.coords 
nprocs = Sf._vector_space.cart.nprocs


for i in range(min(n),2,-1):

    nc = [2**(i-1) +p[0], 2**(i-1) +p[1]]
    Tc = [make_open_knots(p, nc[0]).tolist(), make_open_knots(p, nc[1]).tolist()]

    Sc = Sf.reduce_grid(axes=[0, 1], knots=Tc)
    
    P1   = [None]*ndim
    P2   = [None]*ndim
    R1   = [None]*ndim
    Ac1d = [None]*ndim
    Sc1d = [None]*ndim

    for ax in range(ndim):
    
        Sc1d[ax] = TensorFemSpace(Sc.spaces[ax])
        
        Tc[ax] = Sc.spaces[ax].knots
        nc[ax] = Sc.spaces[ax].nbasis
        Ts     = knots_to_insert(Tf[ax], nf[ax], p[ax], Tc[ax], nc[ax], p[ax])
        P1[ax] = matrix_multi_stages(Ts, nc[ax], p[ax], Tc[ax])*(nc[ax]-p[ax])/(nf[ax]-p[ax])
        P1[ax] = csc_matrix(P1[ax])
        R1[ax] = P1[ax].transpose()
        P2[ax] = csr_matrix(P1[ax])
        
        Vf1d = Sf1d[ax].vector_space
        Vc1d = Sc1d[ax].vector_space
        
        [sf1] = Vf1d.starts
        [ef1] = Vf1d.ends
        [sc1] = Vc1d.starts
        [ec1] = Vc1d.ends

        Ac1d[ax] = StencilMatrix(Vc1d, Vc1d)
        
        M  = stencil_mat_dot_csc(Af1d[ax]._data, P1[ax].data, P1[ax].indices, P1[ax].indptr, p[ax], sf1, ef1, nc[ax])

        csr_dot_csc(R1[ax].data, R1[ax].indices, R1[ax].indptr, M.data, M.indices, M.indptr, 
                    Ac1d[ax]._data , sc1, ec1, nc[ax], sf1, ef1 ,p[ax])

        if coords[ax] == 0:
            Ac1d[ax][sc1,:] = 0.
            Ac1d[ax][sc1,0] = 1.
        if coords[ax] == nprocs[ax]-1:
            Ac1d[ax][ec1,:] = 0
            Ac1d[ax][ec1,0] = 1.

    Ac = KroneckerStencilMatrix_2D(Sc.vector_space, Sc.vector_space, *Ac1d)

    Sf1d = Sc1d
    Sf = Sc

    nf   = nc
    Tf   = Tc
    Af1d = Ac1d
    
    T.append(Tc)
    P.append(P2)
    R.append(R1)
    V.append(Sc.vector_space)
    A.append(Ac)
    starts.append(V[-1].starts)
    ends  .append(V[-1].ends)


xf = StencilVector(V[0])

xf[starts[0][0]:ends[0][0]+1, starts[0][1]:ends[0][1]+1] = 1
                                                           


if coords[0] == 0:
    xf[starts[0][0],:] = 0.
    
if coords[1] == 0:
    xf[:,starts[0][1]] = 0.

if coords[0] == nprocs[0]-1:
    xf[ends[0][0],:] = 0.

if coords[1] == nprocs[1]-1:
    xf[:,ends[0][1]] = 0.

xf.update_ghost_regions()

bf = A[0].dot(xf)

xf[:,:] = 0.
E = [xf]
iter = -1
m  = len(P)
nrmr_sqr = 1
while iter <50 and np.sqrt(nrmr_sqr)>1e-5:
    iter +=1
    E = [E[0]]
    
    E[0] = weighted_jacobi(A[0], bf, x0=E[0] ,maxiter=15)

    rf = bf - A[0].dot(E[0])
    nrmr_sqr = rf.dot(rf)
    print(nrmr_sqr)
    Re = [rf]
    
    for i in range(m-1):
        rc = StencilVector(V[i+1])
        xc = StencilVector(V[i+1])
        # restriction
        rf.update_ghost_regions()
        csr_dot_stencil_vec_2d([R[i][0].data,R[i][1].data], [R[i][0].indices, R[i][1].indices], 
                                [R[i][0].indptr, R[i][1].indptr], rf._data, rc._data, 
                                p, starts[i], starts[i+1],ends[i+1])

         #Smoothening
        if coords[0] == 0:
            rc[starts[i+1][0],:] = 0.
            
        if coords[1] == 0:
            rc[:,starts[i+1][1]] = 0.

        if coords[0] == nprocs[0]-1:
            rc[ends[i+1][0],:] = 0.

        if coords[1] == nprocs[1]-1:
            rc[:,ends[i+1][1]] = 0.
            
        ec  = weighted_jacobi(A[i+1], rc, maxiter=15)
        rc -= A[i+1].dot(ec)

        rf = rc
    
        Re.append(rc)
        E.append(ec)
    
    rc = StencilVector(V[m])
    rf.update_ghost_regions()
    csr_dot_stencil_vec_2d([R[m-1][0].data,R[m-1][1].data], [R[m-1][0].indices, R[m-1][1].indices], 
                            [R[m-1][0].indptr, R[m-1][1].indptr], rf._data, rc._data, 
                            p, starts[m-1], starts[m],ends[m])
                        

 
    if coords[0] == 0:
        rc[starts[m][0],:] = 0.
        
    if coords[1] == 0:
        rc[:,starts[m][1]] = 0.

    if coords[0] == nprocs[0]-1:
        rc[ends[m][0],:] = 0.

    if coords[1] == nprocs[1]-1:
        rc[:,ends[m][1]] = 0.

 
    ec, info = cg(A[m], rc, tol=1e-17, verbose=False)

    E.append(ec)
    Re.append(rc)

    for i in range(m-1,0,-1):

        ef = StencilVector(V[i])

        csr_dot_stencil_vec_2d([P[i][0].data,P[i][1].data], [P[i][0].indices, P[i][1].indices], 
                                [P[i][0].indptr, P[i][1].indptr], E[i+1]._data, ef._data,
                                p, starts[i+1], starts[i],ends[i])
        
        
        E[i] += ef
        E[i].update_ghost_regions()
        E[i]  = weighted_jacobi(A[i], Re[i],x0=E[i] ,maxiter=15)

    ef = StencilVector(V[0])

    csr_dot_stencil_vec_2d([P[0][0].data,P[0][1].data], [P[0][0].indices, P[0][1].indices], 
                            [P[0][0].indptr, P[0][1].indptr], E[1]._data, ef._data,
                            p, starts[1], starts[0],ends[0])
        

    E[0] += ef
    E[0].update_ghost_regions()

E[0], info = pcg(A[0], bf, pc='jacobi',x0=E[0], tol=1e-8, maxiter=1)
print(E[0].toarray())
if rank==0:
    print(info)

