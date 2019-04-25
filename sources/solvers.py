# coding: utf-8
# ... Conjugate Residual Method
import numpy as np
from numpy import zeros
from scipy.sparse        import kron, csc_matrix

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
def matmul(A,P,B,n,p):
    # j1 the last non zero element
    for i in range(p+1):
        j1 = 0 
        j2 = int((i+p)/2) + 1
        
        for j in range(j2-j1):
            v = 0.
            for k in range(i+p+1):
                v += A[i,k] * P[j,k]    
            B[i,j] = v

    for i in range(p+1,n-p):  
        j1 = int((i-1-p)/2)
        j2 = int((i+p)/2) + 1

        for j in range(j2-j1):
            v = 0.
            for k in range(2*p+1):
                v += A[i,i-p+k] * P[j+j1, i-p+k]
            B[i,j+j1] = v

            
    for i in range(n-p, n+1):  
        j1 = int((i-p-1)/2)
        j2 = int(n/2)

        for j in range(j2-j1):
            v = 0.
            for k in range(n-i+p+1):
                v += A[i,i-p+k] * P[j+j1, i-p+k]
                
            B[i,j+j1] = v
            
def matmul2(A, P, B, nf, nc, p):
    #i+p = 2*j
    #i-p = 2*j+2
    for j in range(int(p/2)+1):
        i1 = 0
        i2 = 2*j+2+p+1

        for i in range(2*j+2-p):

            v = 0.
            for k in range(i+p-2*j+1):
                v += A[i,2*j+k-i+p] * P[j,k]
                    
            B[j,i-2*j+p] = v
        
        for i in range(2*j+2-p, 2*j+2+p+1):
            
            v = 0.
            for k in range(2*1+1):
                v += A[i,2*j+k-i+p] * P[j,k]
                    
            B[j,i-2*j+p] = v
            

    for j in range(int(p/2)+1,int((nf-1-p)/2)):
        i1 = 2*j-p
        i2 = 2*j+2+p+1

        for i in range(i1,2*j+2-p):
            v = 0.
            for k in range(i+p-2*j+1):
                v += A[i,2*j+k-i+p] * P[j,k]

            B[j,i-2*j+p] = v
            
        for i in range(2*j+2-p, 2*j+p+1):

            v = 0.
            for k in range(2*1+1):
                v += A[i,2*j+k-i+p] * P[j,k]
                    
            B[j,i-2*j+p] = v
            
                    
        for i in range(2*j+p+1,i2):

            v = 0.
            for k in range(i-p-2*j, 2*1+1):
                v += A[i,2*j+k-i+p] * P[j,k]
                    
            B[j,i-2*j+p] = v

    for j in range(int((nf-1-p)/2),nc): 

        i1 = 2*j-p
        i2 = nf
        
        for i in range(i1,2*j+2-p):
            v = 0.
            for k in range(i+p-2*j+1):
                v += A[i,2*j+k-i+p] * P[j,k]

            B[j,i-2*j+p] = v
            
        for i in range(2*j+2-p,i2):
            v = 0.
            for k in range(2*1+1):
                v += A[i, 2*j+k-i+p] * P[j,k]
            
            B[j,i-2*j+p] = v
            
def matmul3(A, B, C, nf, nc, p):

    #i+p = 2*j
    #i-p = 2*j+2
    for j in range(int(p/2)+1):

        for i in range(j+int(p/2)+1):

            C[i,j-i+p] = sum(A[i,:]*B[j,2*i-2*j+p:2*i+3-2*j+p])
            
        for i in range(j+int(p/2)+1,j+1+int(p/2)+1):
            C[i,j-i+p] = sum(A[i,:2*j+2+p+1-2*i]*B[j,2*i -2*j+p:2*p+3])

 
    for j in range(int(p/2)+1,int((nf-1-p)/2)):
    
        for i in range(j-int(p/2)-1,j-int(p/2)):
            C[i,j-i+p] = sum(A[i,2*j-p-2*i:]*B[j,:2*i+3-2*j+p])

        for i in range(j-int(p/2),j+int(p/2)+1):
            C[i,j-i+p] = sum(A[i,:]*B[j,2*i-2*j+p:2*i+3-2*j+p])
            
        for i in range(j+int(p/2)+1,min(j+1+int(p/2)+1,nc)):
            C[i,j-i+p] = sum(A[i,:2*j+2+p+1-2*i]*B[j,2*i -2*j+p:2*p+3])

    for j in range(int((nf-1-p)/2),nc):

        for i in range(j-int(p/2)-1,j-int(p/2)):
            C[i,j-i+p] = sum(A[i,2*j-p-2*i:]*B[j,:2*i+3-2*j+p])

        for i in range(j-int(p/2),nc):
            C[i,j-i+p] = sum(A[i,:]*B[j,2*i-2*j+p:2*i+3-2*j+p])
            

def get_max_row_span(Arow, Aptr, p, s, e):
    ln = 0
    m1  = 0
    m2  = 0
    for j in range(len(Aptr)-1):
        pt1 = Aptr[j]
        pt2 = Aptr[j+1]
        i1  = Arow[pt1]
        i2  = Arow[pt2-1] + 1
        m1  = min(m1,i1)
        m2  = max(m2,i2-1) 
        ln += max(min(i2+p, e+1+p)-max(i1-p, s-p),0)

    return ln
        
def stencil_mat_dot_csc(A, Bdata, Brow, Bptr, p, s, e, nc):
    ln = get_max_row_span(Brow, Bptr, p, s, e)
    Cdata = zeros(ln)
    Crow  = zeros(ln, dtype='int')
    Cptr  = zeros(nc+1, dtype='int')
    m = 0

    for j in range(nc):
        pt1 = Bptr[j]
        pt2 = Bptr[j+1]
        i1  = Brow[pt1]
        i2  = Brow[pt2-1]

        Cptr[j] = m

        for i in range(max(i1-p,s-p),min(i2+p,e+p)+1):

            row = A[p+i-s,:]
            v = 0.
            for k in range(max(i1,i-p), min(i2,i+p)+1):
                
                v +=row[k-i+p]*Bdata[pt1+k-i1]

            Cdata[m] = v
            Crow[m]  = i-s+p
            m += 1
    Cptr[nc] = m

            #C[i,j-i+p] = v
    return csc_matrix((Cdata,Crow, Cptr),shape=(e-s+1+2*p,nc))
    

def csr_dot_csc(Adata, Arow, Aptr,Bdata, Brow, Bptr, C, s, e, nc, sf1, ef1,p):

    for i in range(s, e+1):
        ptj1 = Aptr[i]
        ptj2 = Aptr[i+1]
        j1  = Arow[ptj1]
        j2  = Arow[ptj2-1]
        for jj in range(2*p+1):
            j = i-p+jj
            
            if j<0 or j>=nc:
                continue
            pti1 = Bptr[j]
            pti2 = Bptr[j+1]
            i1  = Brow[pti1] + sf1 - p
            i2  = Brow[pti2-1] + sf1 - p
            v = 0.

            for k in range(max(i1,j1),min(i2,j2)+1):
                v += Adata[ptj1+k-j1]*Bdata[pti1+k-i1]
            

            C[p+i-s,jj] = v
     
     
def csr_dot_stencil_vec(Adata, Arow, Aptr, x, y, p, s1, s2, e2):

    for i in range(s2,e2+1):
       
        pt1 = Aptr[i]
        pt2 = Aptr[i+1]
        j1  = Arow[pt1]
        j2  = Arow[pt2-1]+1
        y[i-s2+p] = (Adata[pt1:pt2]*x[j1-s1+p:j2-s1+p]).sum()
        
def csr_dot_stencil_vec_2d(Adata, Arow, Aptr, x, y, p, s1, s2, e2):

    for i1 in range(s2[0], e2[0]+1):
        pt1s = Aptr[0][i1]
        pt1e = Aptr[0][i1+1]
        j1s  = Arow[0][pt1s]
        j1e  = Arow[0][pt1e-1]+1
        for i2 in range(s2[1], e2[1]+1):
       
            pt2s = Aptr[0][i1]
            pt2e = Aptr[0][i1+1]
            j2s  = Arow[0][pt2s]
            j2e  = Arow[0][pt2e-1]+1


            y[i1-s2[0]+p[0], i2-s2[1]+p[1]] = (np.outer(Adata[0][pt1s:pt1e], Adata[1][pt2s:pt2e])*
                                               x[j1s-s1[0]+p[0]:j1e-s1[0]+p[0], j2s-s1[1]+p[1]:j2e-s1[1]+p[1]]).sum()        
            
def crl(A, b, x0=None, tol=1e-5, maxiter=1000, verbose=False):
    from math import sqrt

    n = A.shape[0]

    assert( A.shape == (n,n) )
    assert( b.shape == (n, ) )

    # First guess of solution
    if x0 is None:
        x = 0.0 * b.copy()
    else:
        assert( x0.shape == (n,) )
        x = x0.copy()

    # First values
    r = b - A.dot(x)
    p = r.copy()
    q = A.dot(p)
    s = q.copy()

    sr = s.dot(r)

    tol_sqr = tol**2

    if verbose:
        print( "CG solver:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"

    # Iterate to convergence
    for k in range(1, maxiter+1):

        if sr < tol_sqr:
            k -= 1
            break

        alpha  = sr / q.dot(q)
        x  = x + alpha*p
        r  = r - alpha*q

        s = A.dot(r)

        srold = sr
        sr = s.dot(r)

        beta = sr/srold

        p = r + beta*p
        q = s + beta*q

        if verbose:
            print( template.format(k, sqrt(sr)))

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': k, 'success': sr < tol_sqr, 'res_norm': sqrt(sr) }

    return x, info
# ...

# ...
def pcg(A, psolve, b, x0=None, tol=1e-6, maxiter=100, verbose=False):
    from math import sqrt

    n = A.shape[0]

    assert( A.shape == (n,n) )
    assert( b.shape == (n, ) )

    # First guess of solution
    if x0 is None:
        x = 0.0 * b.copy()
    else:
        assert( x0.shape == (n,) )
        x = x0.copy()

    # First values
    r = b - A.dot(x)

    nrmr0 = sqrt(r.dot(r))

    s = psolve(A, r)
    p = s
    sr = s.dot(r)

    if verbose:
        print( "CG solver:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"

    # Iterate to convergence
    for k in range(1, maxiter+1):

        q = A.dot(p)
        alpha  = sr / p.dot(q)

        x  = x + alpha*p
        r  = r - alpha*q

        s = A.dot(r)

        nrmr = r.dot(r)

        if nrmr < tol*nrmr0:
            k -= 1
            break

        s = psolve(A, r)

        srold = sr
        sr = s.dot(r)

        beta = sr/srold

        p = s + beta*p

        if verbose:
            print( template.format(k, sqrt(nrmr)))

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': k, 'success': nrmr < tol*nrmr0, 'res_norm': sqrt(nrmr) }

    return x, info
# ...

# ...
class jacobi:
    def __init__(self, A):
        self.shape = A.shape
        self.A     = A
    def matvec(self, b):
        return jacobi1(self.A, b)
        
class gauss_seidel:
    def __init__(self, A,maxiter=10):
        self.shape = A.shape
        self.A = A
        self.maxiter = maxiter
    def matvec(self, b):
        return gauss_seidel1(self.A,b,self.maxiter)
        
def jacobi1(A, b, maxiter=1,x0=None):

    n = A.shape[0]

    assert(A.shape == (n,n))
    assert(b.shape == (n, ))


    if x0 is None:
        x = b.copy()
    else:
        x = x0
    

    for m in range(maxiter):
        x_new = np.zeros_like(x)
        for i1 in range(1,n-1):
                v = 0.
                
                for k in range(1,i1):
                    v +=A[i1,k]*x[k]
                for k in range(i1+1,n-1):
                    v +=A[i1,k]*x[k]
                x_new[i1] = (b[i1] - v)/A[i1,i1]
        x = x_new
    #...
    return x
        
def gauss_seidel1(A,b,maxiter=100, x0=None):
    if x0 is None:
        x = b.copy()
    else:
        x = x0

    for it_count in range(1, maxiter):
        x_new = np.zeros_like(x)

        for i in range(1,A.shape[0]-1):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.allclose(x, x_new, rtol=1e-17):
            break
        x = x_new
    return x


# ...The weighted Jacobi iterative method
def damped_jacobi(A, b, x0=None, tol=1e-6, maxiter=10, verbose=False):
    from math import sqrt
    from spl.linalg.stencil import StencilVector

    n = A.shape[0]

    assert(A.shape == (n,n))
    assert(b.shape == (n, ))

    V = b.space
    [s1, s2] = V.starts
    [e1, e2] = V.ends

    dr = StencilVector(V)
    dr[:,:] = 0

    if verbose:
        print( "Damped Jacobi method:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"

    #...

    # ... the parameter omega
    omega = 2./3
    # ...

    # First guess of solution
    if x0 is None:
        x = 0.0 * b.copy()
    else:
        assert( x0.shape == (n,) )
        x = x0.copy()

    tol_sqr = tol**2


    # Iterate to convergence
    for k in range(1, maxiter+1):

        r = b - A.dot(x)

        for i1 in range(s1, e1+1):
            for i2 in range(s2, e2+1):
                dr[i1, i2] = omega*r[i1, i2]/A[i1, i2, 0, 0]

        dr.update_ghost_regions()

        x  = x + dr

        nrmr = dr.dot(dr)
        if nrmr < tol_sqr:
            k -= 1
            break


        if verbose:
            print( template.format(k, sqrt(nrmr)))

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': k, 'success': nrmr < tol_sqr, 'res_norm': sqrt(nrmr) }


    return x
# ...

# ...
def pcg_glt(A, M1, M2, b, x0=None, tol=1e-6, maxiter=100, verbose=False):
    from math import sqrt
    from kron_product import kron_solve_par

    n = A.shape[0]

    assert( A.shape == (n,n) )
    assert( b.shape == (n, ) )

    # First guess of solution
    if x0 is None:
        x = 0.0 * b.copy()
    else:
        assert( x0.shape == (n,) )
        x = x0.copy()

    # First values
    r = b - A.dot(x)

    nrmr0 = sqrt(r.dot(r))

    s = kron_solve_par(M2, M1, r)
    p = s
    sr = s.dot(r)

    if verbose:
        print( "CG-GL-GLTT solver:" )
        print( "+---------+---------------------+")
        print( "+ Iter. # | L2-norm of residual |")
        print( "+---------+---------------------+")
        template = "| {:7d} | {:19.2e} |"

    # Iterate to convergence
    for k in range(1, maxiter+1):

        q = A.dot(p)
        alpha  = sr / p.dot(q)

        x  = x + alpha*p
        r  = r - alpha*q

        s = A.dot(r)

        nrmr = r.dot(r)

        if nrmr < tol*nrmr0:
            k -= 1
            break

        s = kron_solve_par(M2, M1, r)

        srold = sr
        sr = s.dot(r)

        beta = sr/srold

        p = s + beta*p

        if verbose:
            print( template.format(k, sqrt(nrmr)))

    if verbose:
        print( "+---------+---------------------+")

    # Convergence information
    info = {'niter': k, 'success': nrmr < tol*nrmr0, 'res_norm': sqrt(nrmr) }

    return x, info
# ...
class Jacobi:

    def __init__(self, A, maxiter=1000,tol=1e-9):
        self.mat = A
        self.shape = A.shape
        self.dtype = A.dtype
        self.maxiter = maxiter
        self.tol = tol

    def matvec(self, b):
    
        MAXITER, TOL = self.maxiter,self.tol
        A = self.mat  
        n = len(b)
        xk = np.ones(shape = n,dtype = self.dtype)
          
        D = sparse.diags(1/A.diagonal(), 0, format = 'csc',)
        L = sparse.tril(A, format = 'csc')
        U = sparse.triu(A, format = 'csc')     
         
        T = -(D)*(L+U)
        c = (D)*b
         
        i = 0
        err = TOL + 1
        while i < MAXITER and err > TOL:
            x = T.dot(xk) + c
            err = np.linalg.norm(x-xk, 1)/np.linalg.norm(x,1)
            xk = x
            i += 1
           
        return xk, i
        
def interpolation_matrix(k, fmt=None):
    """Returns the interpolation matrix between the level k and k-1."""
    n = 2**k - 1
    nl = 2**(k-1) - 1

    R = np.zeros((nl, n))
    b = 0.5*0.5*np.array([1., 2., 1.])

    for i in range(nl):
        j = 2*i
        R[i,j:j+3] = b

    P = 2*R.transpose()

    return P, R

