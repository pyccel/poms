# -*- coding: UTF-8 -*-
import sys
import numpy as np
from mpi4py              import MPI
from scipy.sparse        import kron, csc_matrix
from scipy.sparse.linalg import splu

from spl.linalg.stencil  import StencilMatrix, StencilVector, StencilVectorSpace
from spl.fem.splines     import SplineSpace
from spl.fem.tensor      import TensorFemSpace
from spl.core.interface  import make_open_knots, matrix_multi_stages, collocation_cardinal_splines
from matrix_assembler    import assembly, assembly_seq
from multilevels         import knots_to_insert
from solvers             import pcg, damped_jacobi, pcg_glt
from utils               import  array_to_vect_stencil, array_to_mat_stencil

# ...
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ... Spline degree and number of elements
p  = int(sys.argv[1])
nf = int(sys.argv[2])
nc = 64

# ... Knot vectors
Tc = make_open_knots(p, nc)
Tf = make_open_knots(p, nf)
Ts = knots_to_insert(Tf, nf, p, Tc, nc, p)

T = list(Ts)
for i in range(len(Tc)):
    T.insert(i, Tc[i])
T.sort()

# ... Fine space
S1 = SplineSpace(p, knots=T)
S2 = SplineSpace(p, knots=T)
S  = TensorFemSpace(S1, S2, comm=comm)
S_seq = TensorFemSpace(S1, S2)

V = S.vector_space
s1, s2 = V.starts
e1, e2 = V.ends
n1, n2 = V.npts

# ========== FINE Grid ===========================
# ... The matrix: assembly of the elliptic problem
wt_ass = MPI.Wtime()
Af = assembly(S)
wt_ass = MPI.Wtime() - wt_ass

comm.Barrier()

# ... The RHS
bf = StencilVector(V)

for i1 in range(s1, e1+1):
    for i2 in range(s2, e2+1):
        bf[i1,i2] = 1.
bf.update_ghost_regions()

#bf = Af.dot(x0)

# ========== Inter-grid operations ===============
P1 = matrix_multi_stages(Ts, nc, p, Tc)
R1 = P1.transpose()
R = kron(R1, R1)
P = kron(P1, P1)

# ========== COARSE Grid =========================
# TODO a ameliorer
#V_seq = StencilVectorSpace( [n1,n2], [p,p], [False,False] )
#Af_seq = StencilMatrix(V_seq, V_seq)
#Af_seq = np.zeros((n1, n2))
#Af_loc = Af.toarray()[s1:e1+1, s2:e2+1].copy()
#comm.Allgatherv(Af_loc, Af_seq[s1:e1+1, s2:e2+1])

Af_seq = assembly_seq(S_seq)
Ac = R*Af_seq.tocoo()*P

comm.Barrier()
# ========== Pre Smoothing =======================
wt_pre = MPI.Wtime()
pc = eval('damped_jacobi')
xf, info_pre = pcg(Af, pc, bf, tol=1e-6, maxiter=10, verbose=False)
wt_pre = MPI.Wtime() - wt_pre

#err_pre = xf-x0

# ========== Get residual ========================
rf = bf - Af.dot(xf)
rc = R.dot(rf.toarray())
rc = comm.allreduce(rc, op=MPI.SUM )

# ========== Solver in coarse grid ===============
Ac_op  = splu(csc_matrix(Ac))
xc = Ac_op.solve(rc)

# ========== Correction ==========================
xc_p = P.dot(xc)

Xc_p = np.zeros(V.npts)
for i1 in range(n1):
    for i2 in range(n2):
        Xc_p[i1, i2] = xc_p[i1*n2+i2]

rf = array_to_vect_stencil(V, Xc_p)

xf = xf + rf
xf.update_ghost_regions()

# ========== GLT: Post Smoothing =======================
# collocation_matrix
C1 = collocation_cardinal_splines(p, n1)
C2 = collocation_cardinal_splines(p, n2)
M1 = array_to_mat_stencil(n1, p, C1)
M2 = array_to_mat_stencil(n2, p, C2)

comm.Barrier()

wt_pos = MPI.Wtime()
xf_2, info_pos = pcg_glt(Af, M1, M2, bf, x0=xf, tol=1e-6, maxiter=p+1, verbose=False)
wt_pos = MPI.Wtime() - wt_pos

#err_pos = xf_2-x0

# ========== Print infos =========================
np.set_printoptions(linewidth=100000, precision=4)
for i in range(comm.Get_size()):
    if rank == i:
        print('rank= ', rank, (p, nc, n1))
        print('ASSB: {}'.format(wt_ass))
        print("PRES: ", info_pre, wt_pre)
        print("POST: ", info_pos, wt_pos)
        print('', flush=True)
    comm.Barrier()
 #...
