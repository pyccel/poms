# -*- coding: UTF-8 -*-
import numpy as np
from mpi4py             import MPI
from spl.ddm.cart       import Cart
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix



#from pyccel_functions import kron_dot_pyccel_2d
#from pyccel_functions import kron_solve_serial_pyccel_2d
#from pyccel_functions import kron_solve_par_pyccel_2d
from pyccel_functions import kron_solve_par_bnd_pyccel_2d as kron_solve_par_bnd_pyccel_2d_python
#from pyccel_functions import kron_solve_par_bnd_pyccel_3d
#import kron_solve_par_pyccel_2d
import kron_solve_par_bnd_pyccel_2d
import kron_solve_par_bnd_pyccel_3d



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

   # kron_solve_par_pyccel_2d.mod_kron_solve_par_pyccel_2d(A, B, X._data, Y._data, points, pads,
   #                                                starts, ends, subcoms, sizes[0],
   #                                                disps[0],sizes[1],disps[1])
    return X
    # ...

def kron_solve_par_bnd_2d(A_bnd,la ,ua ,B_bnd, lb, ub, Y, X, with_pycc=False):
    X.update_ghost_regions()
    V = Y.space
    starts  = V.starts
    ends    = V.ends
    pads    = V.pads
    points  = V.npts
    disps   = V.cart.global_starts
    sizes    = [None]*2
    
    sizes[0] = V.cart.global_ends[0] - disps[0] + 1
    sizes[1] = V.cart.global_ends[1] - disps[1] + 1
    
	# ...
    if with_pycc:
        subcoms = np.array([V.cart.subcomm[0].py2f(), V.cart.subcomm[1].py2f()])
        kron_solve_par_bnd_pyccel_2d.mod_kron_solve_par_bnd_pyccel_2d(A_bnd, la, ua, B_bnd, lb, ub, 
                                                                  X._data, Y._data, points, pads,
                                                                  starts, ends, subcoms, sizes[0],
                                                                  disps[0],sizes[1],disps[1])
    else:
        subcoms = np.array([V.cart.subcomm[0], V.cart.subcomm[1]])
        kron_solve_par_bnd_pyccel_2d_python(A_bnd, la, ua, B_bnd, lb, ub, 
                                  X._data, Y._data, points, pads,
                                starts, ends, subcoms, sizes[0],
                                  disps[0],sizes[1],disps[1])
	# ...
    
	#X.update_ghost_regions()
    return X

def kron_solve_par_bnd_3d(A_bnd,la ,ua ,B_bnd, lb, ub, C_bnd, lc, uc, Y, X):

    V = Y.space
    starts  = V.starts
    ends    = V.ends
    pads    = V.pads
    points  = V.npts
    disps   = V.cart.global_starts
    sizes    = [None]*3
    
    sizes[0] = V.cart.global_ends[0] - disps[0] + 1
    sizes[1] = V.cart.global_ends[1] - disps[1] + 1
    sizes[2] = V.cart.global_ends[2] - disps[2] + 1
 
    #subcoms = np.array([V.cart.subcomm[0], V.cart.subcomm[1], V.cart.subcomm[2]])
    subcoms = np.array([V.cart.subcomm[0].py2f(), V.cart.subcomm[1].py2f(), V.cart.subcomm[2].py2f()])
   
    #kron_solve_par_bnd_pyccel_3d(A_bnd, la, ua, B_bnd, lb, ub, C_bnd, lc, uc,
    #                              X._data, Y._data, points, pads, starts, ends, 
    #                              subcoms, sizes[0], disps[0],sizes[1],disps[1], 
    #                              sizes[2], disps[2])
    kron_solve_par_bnd_pyccel_3d.mod_kron_solve_par_bnd_pyccel_3d(A_bnd, la, ua, B_bnd, lb, ub, C_bnd, lc, uc,
                                                                   X._data, Y._data, points, pads, starts, ends, 
                                                                  subcoms, sizes[0], disps[0],sizes[1],disps[1], 
                                                                   sizes[2], disps[2])
    #X.update_ghost_regions()
    return X


# ...
