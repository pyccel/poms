# -*- coding: UTF-8 -*-
from mpi4py             import MPI
from numpy              import zeros, linspace
from spl.linalg.stencil import StencilMatrix
from spl.fem.splines    import SplineSpace
from spl.fem.tensor     import TensorFemSpace

from matrix_assembler   import kernel, assembly

'''
Parallel test of a stifness matrix assembly

To launch, run: mpirun -n 2 python3 tests/test_assembly.py
'''

# ... numbers of elements and degres
p1  = 2  ; p2  = 2
ne1 = 32 ; ne2 = 32
# ...

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print('> Grid   :: [{ne1},{ne2}]'.format(ne1=ne1, ne2=ne2))
    print('> Degree :: [{p1},{p2}]'.format(p1=p1, p2=p2))

grid_1 = linspace(0., 1., ne1+1)
grid_2 = linspace(0., 1., ne2+1)

V1 = SplineSpace(p1, grid=grid_1)
V2 = SplineSpace(p2, grid=grid_2)

V = TensorFemSpace(V1, V2, comm=comm)

# ...
wt = MPI.Wtime()
assembly(V, kernel)
wt = MPI.Wtime() - wt

print('rank: ', rank, '> Elapsed time: {}'.format(wt))
# ...
