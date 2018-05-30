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
    k = 0

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
def pcg(A, psolve, b, x0=None, tol=1e-5, maxiter=1000, verbose=False):
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

    s = psolve(r)

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

        s = psolve(r)

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

# ...
