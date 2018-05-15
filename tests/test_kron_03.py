# -*- coding: UTF-8 -*-
import utils
from kron_product       import kron_solve
from spl.linalg.stencil import StencilVectorSpace, \
                               StencilVector, StencilMatrix
'''
Serial test of solving: (B kron A) X = Y

To launch, run: python3 tests/test_kron_03.py
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
Y = StencilVector(V)
A = StencilMatrix(V1, V1)
B = StencilMatrix(V2, V2)
# ...

# ... Fill in A, B and X
utils.populate_1d_matrix(A, 5.)
utils.populate_1d_matrix(B, 6.)
utils.populate_2d_vector(Y)
# ...

# ..
X = kron_solve(B, A, Y)

X_ref = utils.kron_solve_ref(B, A, Y)

print('A =  \n', A.toarray())
print('B =  \n', B.toarray())
print('X =  \n', X._data)
print('X_ref =  \n', X_ref)
# ...

