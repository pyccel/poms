# -*- coding: UTF-8 -*-
import numpy as np
from scipy.sparse import coo_matrix

# ...
def interpolation_matrix(nf, nc):
    """
    Returns the interpolation matrix between two levels
    """
    # nf = 2*nc
    P = np.zeros((nc, nf))
    b = 0.5 * np.array([1., 2., 1.])

    for i in range(0, nc):
        j = 2*i
        P[i, j:j+3] = b

    R = 0.5 * P.transpose()

    P = coo_matrix(P)
    R = coo_matrix(R)

    return P, R
# ...


