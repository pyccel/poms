# coding: utf-8
from spl.ddm.cart import Cart
from mpi4py import MPI

n1=16
n2=32
p1=1
p2=1

cart = Cart(npts=[n1,n2], pads=[p1,p2], periods=[True, True], reorder=True, comm=MPI.COMM_WORLD)

print('>> ',cart._rank_in_topo, cart.coords)
