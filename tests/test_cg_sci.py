# -*- coding: UTF-8 -*-
from mpi4py             import MPI
from numpy              import zeros, ones, linspace
from numpy.linalg       import norm
from spl.linalg.stencil import StencilMatrix, StencilVector
from spl.fem.splines    import SplineSpace
from spl.fem.tensor     import TensorFemSpace

from scipy.sparse.linalg import cg
from matrix_assembler    import kernel, assembly

'''
Test of scipy cg of Ax=b, A is Laplacian matrix
'''

# ...
def solve_sparse(A, b, tol, maxiter, pc=None, x0=None):
    # ...
    def callback(xk):
        global NUM_ITERS
        NUM_ITERS+=1
    # ...

    # ...
    x,status = cg(A, b, \
                  tol=tol, maxiter=maxiter, \
                  M=pc, callback=callback, x0=x0)
    # ...

    return x,status,NUM_ITERS
# ...

###############################################################################
if __name__ == '__main__':


    global NUM_ITERS
    NUM_ITERS = 0
    # ... Fine Grid: numbers of elements and degres
    p1  = 2 ; p2  = 2
    ne1 = 16 ; ne2 = 16
    # ...

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print('> Grid   :: [{ne1},{ne2}]'.format(ne1=ne1, ne2=ne2))
    print('> Degree :: [{p1},{p2}]'.format(p1=p1, p2=p2))

    grid_1 = linspace(0., 1., ne1+1)
    grid_2 = linspace(0., 1., ne2+1)

    S1 = SplineSpace(p1, grid=grid_1)
    S2 = SplineSpace(p2, grid=grid_2)

    S = TensorFemSpace(S1, S2, comm=comm)
    V = S.vector_space

    s1, s2 = V.starts
    e1, e2 = V.ends


    # ... Weak form: compute the stifeness matrix
    A = assembly(S, kernel)

    x0 = StencilVector(V)

    # Fill in vector with random values, then update ghost regions
    for i1 in range(s1, e1+1):
        for i2 in range(s2, e2+1):
            x0[i1,i2] = 1.
    x0.update_ghost_regions()

    #... Compute matrix-vector product
    b = A.dot(x0)

    tol     = 1.e-12
    maxiter = 10000

    # ... Solve the system
    wt = MPI.Wtime()
    x1, info, n_iter = solve_sparse(A.toarray(), b.toarray(), tol, maxiter)
    wt = MPI.Wtime() - wt

    # ...
    err1 = x1-x0.toarray()

   # ..
    print("n_iter= ", n_iter)
    print("err_norm= ", norm(err1))
    print('elapsed time: {}'.format(wt))
    # ...
