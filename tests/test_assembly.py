# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.linalg.stencil import StencilMatrix
from spl.fem.splines    import SplineSpace
from spl.fem.tensor     import TensorFemSpace

from matrix_assembler   import assembly

'''
Parallel test of a stifness matrix assembly

To launch, run: mpirun -n 2 python3 tests/test_assembly.py
'''

# ... numbers of elements and degres
p1  = 1 ; p2  = 1
ne1 = 16 ; ne2 = 16
# ...

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print('> Grid   :: [{ne1},{ne2}]'.format(ne1=ne1, ne2=ne2))
    print('> Degree :: [{p1},{p2}]'.format(p1=p1, p2=p2))

grid_1 = np.linspace(0., 1., ne1+1)
grid_2 = np.linspace(0., 1., ne2+1)

V1 = SplineSpace(p1, grid=grid_1)
V2 = SplineSpace(p2, grid=grid_2)

V = TensorFemSpace(V1, V2, comm=comm)

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
