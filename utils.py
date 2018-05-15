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

# .. Fill in stencil vector
def populate_2d_vector(X):
    e1 = X.ends[0]
    s1 = X.starts[0]
    e2 = X.ends[1]
    s2 = X.starts[1]

    for i1 in range(s1, e1+1 ):
        for i2 in range(s2, e2+1):
            X[i1,i2] = i1 + i2 + 1
# ...

# ... retrun Y = (B kron A) X
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

    X_arr = zeros((e1+1-s1)*(e2+1-s2))

    # ... TODO improve for parallel testing
    i = 0
    for i2 in range(s2, e2+1):
        for i1 in range(s1, e1+1):
            X_arr[i] = X[i1, i2]
            i += 1
    # ...

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
    Y_arr = zeros((e1+1-s1)*(e2+1-s2))

    # ... TODO improve for parallel testing
    i = 0
    for i2 in range(s2, e2+1):
        for i1 in range(s1, e1+1):
            Y_arr[i] = Y[i1, i2]
            i += 1
    # ...

    C_op  = splu(C)
    X = C_op.solve(Y_arr)

    return X
# ...

