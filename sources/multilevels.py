# -*- coding: UTF-8 -*-
import numpy as np
from scipy.sparse import coo_matrix
from psydac.linalg.stencil import StencilVectorSpace, StencilMatrix

# ...
def knots_to_insert(Tf, nf, pf, Tc, nc, pc):

    current = 0
    ts = np.zeros((nf+pf+1))

    for i in range (pf+1, nf):
        t = Tf[i]
        condition = True
        found = False
        j = pc+1

        while condition:

            if t < Tc[j]:
                condition = False
            elif t == Tc[j]:
                condition = True
                found = True

            j = j + 1
            condition = condition and (j <= nc)

        if ((not condition) and (not found)):
            ts[current] = t
            current = current + 1

    return ts[:current]
# ...

###############################################################################
if __name__ == '__main__':

    from spl.core.interface import make_open_knots
    from spl.core.interface import matrix_multi_stages
    pc = 3
    nc = 8

    pf = 3
    nf = 12

    Tc = make_open_knots(pc, nc)
    Tf = make_open_knots(pf, nf)

    Ts =  knots_to_insert(Tf, nf, pf, Tc, nc, pc)

    np.set_printoptions(linewidth=10000, precision=2)
    print(Tc)
    print("\n", Tf)
    print("\n",Ts)


    M = matrix_multi_stages(Ts, nc, pc, Tc)
    P1 = M.transpose()

    print("\n",np.shape(P1), "\n", P1)
