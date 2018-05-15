# -*- coding: UTF-8 -*-
import utils
from kron_product       import kron_dot, kron_solve
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix
'''
Serial test of:
    1) compute: Y = (B kron A) X0
    3) solve:   (B kron A)X = Y
    3) verify: X = X0

To launch, run: python3 tests/test_kron_04.py
'''

# ... numbers of elements and degres
n1 = 8 ; n2 = 4
p1 = 2 ; p2 = 1
# ...

# ... Vector Spaces
V = StencilVectorSpace([n1, n2], [p1, p2], [False, False])
V1 = StencilVectorSpace([n1], [p1], [False])
V2 = StencilVectorSpace([n2], [p2], [False])

# ... Inputs
X0 = StencilVector(V)
A = StencilMatrix(V1, V1)
B = StencilMatrix(V2, V2)
# ...

# ... Fill in A, B and X
utils.populate_1d_matrix(A, 5.)
utils.populate_1d_matrix(B, 6.)
utils.populate_2d_vector(X0)
# ...

# ..

Y = kron_dot(B, A, X0)
X = kron_solve(B, A, Y)


print('X0 =  \n', X0._data)
print('X  =  \n', X._data)
# ...

