# coding: utf-8
# ... Conjugate Residual Method
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
def pcg(A, psolve, b, x0=None, tol=1e-5, maxiter=10, verbose=False):
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
def jacobi(A, b):

    from spl.linalg.stencil import StencilVector
    n = A.shape[0]

    assert(A.shape == (n,n))
    assert(b.shape == (n, ))

    V = b.space

    [s1, s2] = V.starts
    [e1, e2] = V.ends
    [p1, p2] = V.pads

    x = StencilVector(V)
    x[:,:] = 0.

    # ...
    for i1 in range(s1, e1+1):
        for i2 in range(s2, e2+1):
            x[i1, i2] = A[i1, i2, 0, 0]
            x[i1, i2] = b[i1, i2]/ x[i1, i2]
    #...
    x.update_ghost_regions()
    return x
# ...

# ...The weighted Jacobi iterative method
def damped_jacobi(A, b, x0=None, tol=1e-5, maxiter=100, verbose=False):
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

