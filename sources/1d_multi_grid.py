# -*- coding: UTF-8 -*-
import sys
import numpy as np
from mpi4py              import MPI
from scipy.sparse        import kron, csc_matrix,coo_matrix
from scipy.sparse.linalg import splu

from spl.linalg.stencil  import StencilMatrix, StencilVector, StencilVectorSpace
from spl.fem.splines     import SplineSpace
from spl.fem.tensor      import TensorFemSpace
from spl.core.interface  import make_open_knots, matrix_multi_stages, collocation_cardinal_splines
from matrix_assembler    import assembly_1d
from multilevels         import knots_to_insert
from scipy.sparse.linalg import cg
from solvers import Jacobi,matmul,matmul2,matmul3
np.set_printoptions(precision=6)
# ...
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ... Spline degree and number of elements


p  = 2
nf = 2**2


# Create uniform grid
grid = np.linspace( 0.,1., num=nf+1)

V = SplineSpace( p, grid=grid, periodic=False )
 
Vf  = TensorFemSpace(V, comm=comm)
Vf.init_fem()
V  = Vf.vector_space


# ========== FINE Grid ===========================
# ... The matrix: assembly of the elliptic problem

A = assembly_1d(Vf)
A = A
# ... The RHS
b = StencilVector(V)
b[:] = 1.
b = b.toarray()
Af = A
bf = b
nc = nf
print(A.toarray().shape)
while nc>2:

    nc =  int(nc/2)
    n = nc + 1 -2
    data = np.array([1., 2., 1.]*n)
    rows = np.zeros(3*n,'int')
    cols = rows.copy()
    for i in range(n):
        rows[3*i:3*i+3] = i
    for i in range(n):
        cols[3*i:3*i+3] = (2*i,2*i+1,2*i+2)


    P   = data.reshape((n,3))
    P1 = coo_matrix((data,(rows, cols)))
    print(P1.toarray())
    print(P1.toarray().shape)
    raise SystemExit()
#    R1 = P1.toarray()
#    P1 = R1.T.copy()
#    B1 = np.zeros((np.size(A,0), np.size(P,1)))
    B = np.zeros((n, 2*p+3))
    matmul2(Af._data, P, B, A.shape[0], n, p)




#    C1 = R1.dot(B1)
    C = np.zeros((n, 2*p+1))

    matmul3(P, B, C, A.shape[0], n, p)
    #print(R1)
    #print(B1)
    #print(P)
    #print(B)
    
    print(C)



    nc +=1
    raise SystemExit()
if True:
# ========== COARSE Grid =========================

    Ac = R.dot(Af.dot(P))

# ========== Pre Smoothing =======================

    xf, info_pre = cg(Af, bf, x0=xf, tol=1e-9, maxiter=3)

# ========== Get residual ========================
    rf = bf - Af.dot(xf)
    rc = R.dot(rf)

# ========== Solver in coarse grid ===============
    ec, info_pre = cg(Ac, rc, tol=1e-9, maxiter=3)
    

# ========== Correction ==========================
    ef = P.dot(ec)

    ef, info_pre = cg(Af, rf, x0=ef, tol=1e-9, maxiter=10)
    xf += ef


xect= np.linalg.inv(Af.toarray()).dot(bf)
print(xect-xf)

