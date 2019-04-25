# -*- coding: UTF-8 -*-
import sys
import numpy as np
from mpi4py              import MPI
from scipy.sparse        import kron, csc_matrix,csr_matrix
from scipy.sparse.linalg import splu

from psydac.linalg.stencil  import StencilMatrix, StencilVector, StencilVectorSpace
from psydac.fem.splines     import SplineSpace
from psydac.fem.tensor      import TensorFemSpace
from psydac.core.interface  import make_open_knots, matrix_multi_stages, collocation_cardinal_splines
from matrix_assembler    import assembly_2d_seq
from multilevels         import knots_to_insert
from solvers             import pcg, damped_jacobi, pcg_glt, stencil_mat_dot_csc, csr_dot_csc, csr_dot_stencil_vec
from utils               import  array_to_vect_stencil, array_to_mat_stencil
from matrix_assembler    import assembly_1d
from psydac.linalg.iterative_solvers import weighted_jacobi,cg,pcg
import time
np.set_printoptions(linewidth=1000, precision=4)

# ...
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# ... Spline degree and number of elements
p  = 2
n  = 3
nf = 2**n +p

maxiter = 100

# ... Knot vectors
Tf = make_open_knots(p, nf)

T      = [Tf]
P      = []
R      = []
A      = []
V      = []
starts = []
ends   = []

Sf = SplineSpace(p, knots=Tf)
Sf  = TensorFemSpace(Sf, comm=comm)
    
Af, mass = assembly_1d(Sf)

Af.update_ghost_regions()
A.append(Af)
V.append(Sf.vector_space)
starts.append(V[0].starts[0])
ends  .append(V[0].ends[0])
Sc = Sf

if rank == 0:
    Af[0,:] = 0
    Af[0,0] = 1.
if rank == size-1:
    Af[ends[0],:] = 0
    Af[ends[0],0] = 1.
    
for i in range(n,2,-1):

    nc = 2**(i-1) +p
    Tc = make_open_knots(p, nc).tolist()
    Sc = Sc.reduce_grid(axes=[0], knots=[Tc])
    nc = Sc.spaces[0].nbasis
    Tc = Sc.spaces[0].knots
    Ts = knots_to_insert(Tf, nf, p, Tc, nc, p)
    
    P1 = matrix_multi_stages(Ts, nc, p, Tc)*(nc-p)/(nf-p)
    P1 = csc_matrix(P1)
    R1 = P1.transpose()
    P2 = csr_matrix(P1)
    
    Vf = Sf.vector_space
    Vc = Sc.vector_space

    Ac = StencilMatrix(Vc, Vc)

    [sf1] = Vf.starts
    [ef1] = Vf.ends
    [sc1] = Vc.starts
    [ec1] = Vc.ends

        
    M  = stencil_mat_dot_csc(Af._data, P1.data, P1.indices, P1.indptr, p, sf1, ef1, nc)


    csr_dot_csc(R1.data, R1.indices, R1.indptr, M.data, M.indices, M.indptr, Ac._data , sc1, ec1, nc, sf1, ef1 ,p)
    Ac.update_ghost_regions()

    if rank == 0:
        Ac[sc1,:] = 0.
        Ac[sc1,0] = 1.
    if rank == size-1:
        Ac[ec1,:] = 0
        Ac[ec1,0] = 1.
        
    Sf = Sc
    nf = nc
    Tf = Tc
    Af = Ac
    T.append(Tc)
    P.append(P2)
    R.append(R1)
    V.append(Vc)
    A.append(Ac)
    starts.append(Vc.starts[0])
    ends  .append(Vc.ends[0])

print(starts)
print(ends)
xf = StencilVector(V[0])
xf[starts[0]:ends[0]+1] = 1

if rank == 0:
    xf[starts[0]] = 0.
if rank == size-1:
    xf[ends[0]] = 0.

xf.update_ghost_regions()

bf = A[0].dot(xf)
xf[:] = 0.


E = [xf]
iter = -1
m  = len(P)
nrmr_sqr = 1
a = time.time()
while iter <20 and np.sqrt(nrmr_sqr)>1e-5 :
    iter +=1
    E = [E[0]]
    
    E[0] = weighted_jacobi(A[0], bf,x0=E[0] ,maxiter=p)

    rf = bf - A[0].dot(E[0])
    nrmr_sqr = rf.dot(rf)
    Re = [rf]
    
    
    for i in range(m-1):
        rc = StencilVector(V[i+1])
        xc = StencilVector(V[i+1])
        # restriction
        rf.update_ghost_regions()
        csr_dot_stencil_vec(R[i].data, R[i].indices, R[i].indptr, rf._data, rc._data, 
                            p, starts[i], starts[i+1],ends[i+1])
         #Smoothening
        if rank == 0:
            rc[starts[i+1]] = 0
        if rank == size -1 :
            rc[ends[i+1]] = 0
            
        ec  = weighted_jacobi(A[i+1], rc, maxiter=p)
        rc -= A[i+1].dot(ec)
        rf = rc
        
        
        
        Re.append(rc)
        E.append(ec)
    
    rc = StencilVector(V[m])
    rf.update_ghost_regions()
    csr_dot_stencil_vec(R[m-1].data, R[m-1].indices, R[m-1].indptr, rf._data, rc._data, 
                        p, starts[m-1], starts[m],ends[m])
                        
                  
    if rank == 0:
        rc[starts[m]] = 0
    if rank == size -1 :
        rc[ends[m]] = 0

    ec, info = cg(A[m], rc, tol=1e-17, verbose=False)

    E.append(ec)
    Re.append(rc)
    
    for i in range(m-1,0,-1):

        ef = StencilVector(V[i])

        csr_dot_stencil_vec(P[i].data, P[i].indices, P[i].indptr, E[i+1]._data, ef._data,
        p, starts[i+1], starts[i],ends[i])
        
        
        E[i] += ef
        E[i].update_ghost_regions()
        E[i]  = weighted_jacobi(A[i], Re[i],x0=E[i] ,maxiter=p)

    ef = StencilVector(V[0])

    csr_dot_stencil_vec(P[0].data, P[0].indices, P[0].indptr, E[1]._data, ef._data,
                        p, starts[1], starts[0],ends[0])
        

    E[0] += ef
    E[0].update_ghost_regions()
    
print(E[0].toarray())
raise SystemExit()
E[0], info = pcg(A[0], bf, pc='jacobi',x0=E[0]  , tol=1e-8)
b = time.time()

E[0], info2 = pcg(A[0], bf, pc='jacobi', tol=1e-8)
b2 = time.time()
if rank==0:
    print(iter)
    print(np.sqrt(nrmr_sqr))
    print(info)
    print(info2)
    print(b-a)
    print(b2-b)

