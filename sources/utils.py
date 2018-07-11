# -*- coding: UTF-8 -*-
"""
  Some utility functions useful for testing
"""

# .. Fill in stencil matrix as band
def populate_1d_matrix(M, diag):
    e = M.ends[0]
    s = M.starts[0]
    p = M.pads[0]

    for i in range(s, e+1):
        for k in range(-p, p+1):
            M[i, k] = k
        M[i, 0] = diag
    M.remove_spurious_entries()
# ...

# ...
def populate_2d_matrix(M, diag):
    p1 = M.pads[0]
    p2 = M.pads[1]

    for k1 in range(-p1,p1+1):
        for k2 in range(-p2,p2+1):
            M[:,:,k1,k2] = 10.- abs(k1)-abs(k2)
    M.remove_spurious_entries()
# ...

# .. Fill in stencil vector
def populate_2d_vector(X):
    e1 = X.ends[0]
    s1 = X.starts[0]
    e2 = X.ends[1]
    s2 = X.starts[1]

    for i1 in range(s1, e1+1 ):
        for i2 in range(s2, e2+1):
            X[i1,i2] = 1.
# ...

# ... retrun Y = (B kron A) Xt
def kron_dot_ref(B, A, X):
    from numpy import zeros
    from scipy.sparse import kron

    # ...
    A_csr = A.tocsr()
    B_csr = B.tocsr()
    C = kron(B_csr, A_csr)

    # ...
    V = X.space

    [s1, s2] = V.starts
    [e1, e2] = V.ends
    [p1, p2] = V.pads

    X_arr = X.toarray()
    Y = C.dot(X_arr)

    return  Y
# ...

# ... return X, solution of (B kron A)X = Y
def kron_solve_ref(B, A, Y):
    from numpy import zeros
    from scipy.sparse import csc_matrix, kron
    from scipy.sparse.linalg import splu

    # ...
    A_csr = A.tocsr()
    B_csr = B.tocsr()
    C = csc_matrix(kron(B_csr, A_csr))

    # ...
    V = Y.space

    [s1, s2] = V.starts
    [e1, e2] = V.ends
    [p1, p2] = V.pads

    # ...
    Y_arr = Y.toarray()
    C_op  = splu(C)
    X = C_op.solve(Y_arr)

    return X
# ...

# ... v_arr: 2d numpy array
def array_to_vect_stencil(v_space, v_arr):
    from spl.linalg.stencil import StencilVector

    v_stencil = StencilVector(v_space)

    idx_to  = tuple( slice(p,-p) for p in v_space.pads )
    idx_arr = tuple( slice(s,e+1) for s,e in zip(v_space.starts, v_space.ends) )
    v_stencil._data[idx_to] = v_arr[idx_arr]

    return v_stencil
# ...

# ... return matrix stencil 1d
def array_to_mat_stencil(n, p, v_arr):
    from spl.linalg.stencil import StencilMatrix,StencilVectorSpace
    from spl.core.interface import make_open_knots, compute_spans

    V = StencilVectorSpace([n], [p], [False])
    m_stencil = StencilMatrix(V, V)

    n = V.npts[0]
    s = V.starts[0]
    e = V.ends[0]
    p = V.pads[0]

    T = make_open_knots(p, n)
    spans = compute_spans(p, n, T)

    for ie in range(s, e):
        i = spans[ie]
        k  = i - p - 1
        ks = max(0, s-k)
        ke = min(p, e-k)

        for il in range(ks, ke+1):
            i1 = i - p  - 1 + il

            for jl in range(0, p+1):
                j1   = i - p  - 1 + jl
                m_stencil[i1, j1 - i1] =  v_arr[i1, j1]


    return m_stencil
# ...
