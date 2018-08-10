# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix



#from pyccel_functions import kron_dot_pyccel
#from pyccel_functions import kron_solve_serial_pyccel
#from pyccel_functions import kron_solve_par_pyccel
import kron_solve_par_pyccel, kron_solve_par_bnd_pyccel



def _update_ghost_regions_parallel( u, comm_cart, direction, shift_info,recv_types,send_types):

    requests = [0]*4
    i = 0
    for disp in range(-1,2,2):
        ind = int(disp/2+0.5)
        info_rank_source = shift_info[ direction, ind, 0]
        recv_type = recv_types[direction][ind]
        recv_req = comm_cart.Irecv((u, 1, recv_type), info_rank_source, disp+42 )
        requests[i] = recv_req
        i +=1

    for disp in range(-1,2,2):
        ind = int(disp/2+0.5)
        info_rank_dest  = shift_info[direction, ind, 1]
        send_type = send_types[direction][ind]
        send_req = comm_cart.Isend( (u, 1, send_type), info_rank_dest, disp+42 )
        requests[i] = send_req
        i +=1

    MPI.Request.Waitall( requests )


def kron_dot_v2(A, B, X):
    # ...
    V = X.space

    starts = V.starts
    ends = V.ends
    pads = V.pads

    Y = StencilVector(V)
    Y._data[:,:] = 0.


    space     = X._space
    cart      = space.cart
    comm_cart = cart.comm_cart


    shift_info = np.zeros((2,2,2))

    send_types = [[None,None],[None,None]]
    recv_types = [[None,None],[None,None]]

    for disp in range(-1,2,2):
        for direction in range(2):

            info = cart.get_shift_info( direction, disp )
            rank_source = info['rank_source']
            rank_dest = info['rank_dest']
            i = disp if disp>0 else disp+1
            shift_info[direction, i,0] = rank_source
            shift_info[direction, i,1] = rank_dest
            send_types[direction][i]=space.get_send_type(direction, disp)
            recv_types[direction][i]=space.get_recv_type(direction, disp)


    X_tmp = StencilVector(V)

    _update_ghost_regions_parallel(X._data,comm_cart,0,shift_info,recv_types,send_types)
    _update_ghost_regions_parallel(X._data,comm_cart,1,shift_info,recv_types,send_types)

    kron_dot_pyccel(starts, ends, pads, X._data.T, X_tmp._data.T, Y._data.T, A._data.T, B._data.T)

    _update_ghost_regions_parallel(Y._data,comm_cart,0,shift_info,recv_types,send_types)
    _update_ghost_regions_parallel(Y._data,comm_cart,1,shift_info,recv_types,send_types)

    return Y



def kron_solve_serial(A, B, Y):

    V = Y.space
    X = StencilVector(V)
    points = V.npts
    pads   = V.pads

    A = A.toarray().copy(order = 'F')
    B = B.toarray().copy(order = 'F')
    X._data, Y._data = X._data.copy(order = 'F'), Y._data.copy(order = 'F')

    kron_solve_serial_pyccel(B, A, X._data, Y._data, points, pads)

    return X

# ...

# ... Compute X, solution of (B kron A)X = Y
# ... Parallel Version
def kron_solve_par(A, B, Y):

    V = Y.space
    X = StencilVector(V)
    starts  = V.starts
    ends    = V.ends
    pads    = V.pads
    points  = V.npts
    disps   = V.cart.global_starts
    sizes    = [None]*2
    for i in range(2):
        sizes[i] = V.cart.global_ends[i] - disps[i] + 1

    subcoms = np.array([V.cart.subcomm[0].py2f(), V.cart.subcomm[1].py2f()])
    X._data = X._data.copy(order = 'F')
    Y._data = Y._data.copy(order = 'F')

    kron_solve_par_pyccel.mod_kron_solve_par_pyccel(A, B, X._data, Y._data, points, pads,
                                                   starts, ends, subcoms, sizes[0],
                                                   disps[0],sizes[1],disps[1])
    return X
    # ...

def kron_solve_par_bnd(A_bnd,la ,ua ,B_bnd, lb, ub, Y):

    V = Y.space
    X = StencilVector(V)
    starts  = V.starts
    ends    = V.ends
    pads    = V.pads
    points  = V.npts
    disps   = V.cart.global_starts
    sizes    = [None]*2
    for i in range(2):
        sizes[i] = V.cart.global_ends[i] - disps[i] + 1

    subcoms = np.array([V.cart.subcomm[0].py2f(), V.cart.subcomm[1].py2f()])
    X._data = X._data.copy(order = 'F')
    Y._data = Y._data.copy(order = 'F')

    kron_solve_par_bnd_pyccel.mod_kron_solve_par_bnd_pyccel(A_bnd, la, ua, B_bnd, lb, ub, 
                                                            X._data, Y._data, points, pads,
                                                            starts, ends, subcoms, sizes[0],
                                                            disps[0],sizes[1],disps[1])

    return X

# ...
