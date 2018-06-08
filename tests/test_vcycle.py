# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.linalg.stencil import StencilMatrix
from spl.fem.splines    import SplineSpace
from spl.fem.tensor     import TensorFemSpace

from matrix_assembler   import assembly

# ... Spline degree
p = 1

# ... Coarse and fine grids
ne_c = 4
ne_f = 8

# ...
Tc = make_open_knots(p, n1)
Tf = make_open_knots(p, n1)

Ts =  knots_to_insert(Tf, nf, pf, Tc, nc, pc)

# ...
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


grid_1 = np.linspace(0., 1., ne1)
grid_2 = np.linspace(0., 1., ne2)

grid_c = np.linspace(0., 1., nec)

Vc = SplineSpace(p1, grid=grid_c)
V1 = SplineSpace(p1, grid=grid_1)
V2 = SplineSpace(p2, grid=grid_2)

V = TensorFemSpace(V1, V2, comm=comm)
print(">>> KNOTS F: ", V1.knots)
print(">>> KNOTS C: ", Vc.knots)
# ...
wt = MPI.Wtime()
M  = assembly(V)
wt = MPI.Wtime() - wt

np.set_printoptions(linewidth=10000, precision=2)

# ..
for i in range(comm.Get_size()):
    if rank == i:
        print('rank= ', rank)
#        print(M.toarray())
        print(M.shape)
        print('Elapsed time: {}'.format(wt))
        print('', flush=True)
    comm.Barrier()
# ...
