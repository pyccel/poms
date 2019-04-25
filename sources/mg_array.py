# -*- coding: UTF-8 -*-
import sys
import numpy as np
from mpi4py              import MPI
from scipy.sparse        import kron, csc_matrix,csr_matrix
from scipy.sparse.linalg import splu
from spl.linalg.stencil  import StencilMatrix, StencilVector, StencilVectorSpace
from spl.fem.splines     import SplineSpace
from spl.fem.tensor      import TensorFemSpace
from spl.core.interface  import make_open_knots, matrix_multi_stages, collocation_cardinal_splines
from matrix_assembler    import assembly_2d_seq
from multilevels         import knots_to_insert
from solvers             import pcg, damped_jacobi, pcg_glt, stencil_mat_dot_csc, csr_dot_csc, csr_dot_stencil_vec,interpolation_matrix
from utils               import  array_to_vect_stencil, array_to_mat_stencil
from matrix_assembler    import assembly_1d,assemble_rhs
from scipy.sparse.linalg import cg
from solvers import jacobi,jacobi1,gauss_seidel,gauss_seidel1

np.set_printoptions(linewidth=1000, precision=4)

# ...
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# ... Spline degree and number of elements
p  = 11
n  = 4
nf = 2**n + p
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

boundries = Sf.boundries[0]

    
Af, mass = assembly_1d(Sf)
Af = Af.toarray()

bf = np.array(range(len(Af)))
bf[0] = 0
bf[-1] = 0

A.append(Af)
Af[0]=0
Af[-1]=0
bf = Af.dot(bf)


Sc = Sf
E  = []

for i in range(n,2,-1):

    nc = 2**(i-1) +p
    Tc = make_open_knots(p, nc)
    for b in boundries:
        if b not in Tc:
            Tc.append(b)
            nc += 1
    Tc.sort()
    Ts = knots_to_insert(Tf, nf, p, Tc, nc, p)
    
    P1 = matrix_multi_stages(Ts, nc, p, Tc)
    R1 = P1.transpose()*(nc-p)/(nf-p)
    
    #P1, R1 = interpolation_matrix(n)
    M  = Af.dot(P1)
    Ac = R1.dot(M)
    
    nf = nc
    Tf = Tc
    Af = Ac
    Ac[0] = 0
    Ac[-1] = 0

    
    P.append(P1)
    R.append(R1)
    A.append(Ac)
    E.append(np.zeros(Ac.shape[0]))


x,info = cg(A[0], bf,tol=1e-17)
E.insert(0,np.zeros(len(bf)))

m  = len(P)
iter = -1
print(m)

while ((E[0]-x)**2).sum()>1e-8 and iter<300:
    print(E[0])
    E[0]  = gauss_seidel1(A[0], bf, x0=E[0],maxiter=p)
    rf = bf - A[0].dot(E[0])
    Re = [rf]
    iter +=1
    
    for i in range(m-1):
        
        rc = R[i].dot(Re[-1])
        rc[0] = 0
        rc[-1] = 0

        E[i+1]  = gauss_seidel1(A[i+1], rc, maxiter=p)

        rc -= A[i+1].dot(E[i+1])
        Re.append(rc)


    rc = R[m-1].dot(Re[m-1])
    rc[0] = 0
    rc[-1] = 0
    E[m], info = cg(A[m], rc, tol=1e-17)

    
    for i in range(m-1,0,-1):

        ef     = P[i].dot(E[i+1])

        E[i] += ef

        E[i]  = gauss_seidel1(A[i], Re[i], x0=E[i] , maxiter=p)


        assert ef[0]==0 and ef[-1] == 0
    ef = P[0].dot(E[1])
    assert ef[0]==0 and ef[-1] == 0
    print(((E[0]-x)**2).sum())   
    E[0] += ef
print(((E[0]-x)**2).sum())


