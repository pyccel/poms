# -*- coding: UTF-8 -*-
import utils
from kron_product       import kron_dot
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix
'''
Serial test of kron_dot: Y = (B kron A) X

To launch, run: python3 tests/test_kron_01.py
'''

# ... numbers of elements and degres
n1 = 8 ; n2 = 4
p1 = 3 ; p2 = 1
# ...

# ... Vector Spaces
V = StencilVectorSpace([n1, n2], [p1, p2], [False, False])
V1 = StencilVectorSpace([n1], [p1], [False])
V2 = StencilVectorSpace([n2], [p2], [False])

# ... Inputs
X = StencilVector(V)
A = StencilMatrix(V1, V1)
B = StencilMatrix(V2, V2)
# ...

# ... Fill in A, B and X
utils.populate_1d_matrix(A, 5.)
utils.populate_1d_matrix(B, 6.)
utils.populate_2d_vector(X)
# ...

# ..
Y = kron_dot(B, A, X)
Y_ref = utils.kron_dot_ref(B, A, X)

print('Y  = \n', Y._data)
print('Y_ref = \n', Y_ref)
# ...

