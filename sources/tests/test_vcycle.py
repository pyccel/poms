# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.linalg.stencil import StencilMatrix
from spl.fem.splines    import SplineSpace
from spl.fem.tensor     import TensorFemSpace
from spl.core.interface import make_open_knots, matrix_multi_stages
from matrix_assembler   import assembly
from multilevels        import knots_to_insert
from scipy.sparse       import coo_matrix, kron
# ... Spline degree
p = 1

# ... Coarse and fine grids
nc = 8
nf = 10

# ...
Tc = make_open_knots(p, nc)
Tf = make_open_knots(p, nf)

Ts =  knots_to_insert(Tf, nf, p, Tc, nc, p)

T = list(Ts)
for i in range(len(Tc)):
    T.insert(i, Tc[i])
T.sort()

# ...
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


V1 = SplineSpace(p, knots=T)
V2 = SplineSpace(p, knots=T)

Vh = TensorFemSpace(V1, V2, comm=comm)

# ...
wt = MPI.Wtime()
Ah = assembly(Vh)
wt = MPI.Wtime() - wt

P1 = matrix_multi_stages(Ts, nc, p, Tc)
R1 = P1.transpose()
R = kron(R1, R1)
P = kron(P1, P1)

Ah_coo = Ah.tocoo()

np.set_printoptions(linewidth=10000, precision=2)

AH = R*Ah_coo*P
# ..
for i in range(comm.Get_size()):
    if rank == i:
        print('rank= ', rank)
#        print(M.toarray())
        print("Ah:\t", np.shape(Ah_coo))
        print("P:\t",np.shape(P))
        print("R:\t",np.shape(R))
        print("AH:\t", np.shape(AH))
        print('Elapsed time: {}'.format(wt))
        print('', flush=True)
    comm.Barrier()
# ...

