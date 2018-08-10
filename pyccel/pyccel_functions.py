from pyccel import epyccel

header = '#$ header function  kron_dot_pyccel(int[:], int[:], int[:], double[:,:], double[:,:], double[:,:], double[:,:], double[:,:])'
def kron_dot_pyccel(starts, ends, pads, X, X_tmp, Y, A, B):
    s1 = starts[0]
    s2 = starts[1]
    e1 = ends[0]
    e2 = ends[1]
    p1 = pads[0]
    p2 = pads[1]

    for j1 in range(s1-p1, e1+p1+1):
        for i2 in range(s2, e2+1):
             
             X_tmp[j1+p1-s1, i2-s2+p2] = sum(X[j1+p1-s1, i2-s2+k]*B[i2,k] for k in range(2*p2+1))
    
    for i1 in range(s1, e1+1):
        for i2 in range(s2, e2+1):
             Y[i1-s1+p1,i2-s2+p2] = sum(A[i1, k]*X_tmp[i1-s1+k, i2-s2+p2] for k in range(2*p1+1))

    return Y

kron_dot_pyccel = epyccel(kron_dot_pyccel, header)

header='#$ header function kron_solve_serial_pyccel(double[:,:](order = F),double[:,:](order = F),double[:,:](order = F),double[:,:](order = F),int[:], int[:])'
def kron_solve_serial_pyccel(A, B, X, Y, points, pads):
    
    from scipy.linalg.lapack import dgetrf, dgetrs
    from numpy import zeros
    
    n1 = points[0]
    n2 = points[1]
    p1 = pads[0]
    p2 = pads[1]
    X_tmp  = zeros((n2, n1))
    B_finfo = -1
    A_finfo = -1
    A_sinfo = -1
    B_sinfo = -1
    A_piv   = zeros(min(n1,n2),'int')
    B_piv   = zeros(min(n1,n2),'int')
    A, A_piv, A_finfo = dgetrf(A)
    B, B_piv, B_finfo = dgetrf(B)

    for i2 in range(n2):
        X_tmp[i2, 0:n1] = Y[p1:n1+p1, i2 +p2]
        X_tmp[i2, 0:n1], A_sinfo = dgetrs(A, A_piv, X_tmp[i2, 0:n1])

    for i1 in range(n1):
        X[i1+p1, p2:n2+p2] = X_tmp[0:n2, i1]
        X[i1+p1, p2:n2+p2], B_sinfo = dgetrs(B, B_piv, X[i1+p1, p2:n2+p2])

    return X

kron_solve_serial_pyccel = epyccel(kron_solve_serial_pyccel, header, libs = ['lapack'])


header ='#$ header function kron_solve_par_pyccel(double[:,:](order=F), double[:,:](order=F), double[:,:](order=F), double[:,:](order=F) ,int[:], int[:], int[:], int[:], int[:],int[:], int[:],int[:],int[:])'
def kron_solve_par_pyccel(A, B, X, Y, points, pads, starts, ends, subcoms, size_0, disp_0, size_1, disp_1):
    
    from mpi4py import MPI
    from scipy.linalg.lapack import dgetrf, dgetrs
    from numpy import zeros, zeros_like
    ierr = -1
   
    
    subcomm_1 = subcoms[0]
    subcomm_2 = subcoms[1]

    s1 = starts[0]
    s2 = starts[1]
    e1 = ends[0]
    e2 = ends[1]
    n1 = points[0]
    n2 = points[1]
    p1 = pads[0]
    p2 = pads[1]
  
    Y_glob_1 = zeros((n1))
    Ytmp_glob_1 = zeros((n1, e2-s2+1),dtype = 'double',order = 'F')
    Ytmp_glob_2 = zeros((n2), dtype = 'double',order = 'F')
    X_glob_2 = zeros((e1-s1+1, n2),dtype = 'double',order = 'F')
   
    B_finfo = -1
    A_finfo = -1
    A_sinfo = -1
    B_sinfo = -1
    A_piv   = zeros(min(n1,n2),dtype = 'int',order = 'F')
    B_piv   = zeros(min(n1,n2),dtype = 'int',order = 'F')
    A, A_piv, A_finfo = dgetrf(A)
    B, B_piv, B_finfo = dgetrf(B)
    Y_loc = zeros(e1-s1+1)
    Ytmp_loc = zeros(e2-s2+1)

    for i2 in range(e2-s2+1):
        Y_loc[:] = Y[p1:e1-s1+p1+1, i2+p2]
        subcomm_1.Allgatherv(Y_loc, [Y_glob_1, size_0, disp_0])
        Ytmp_glob_1[:,i2] = Y_glob_1[:] 
        Ytmp_glob_1[:,i2], A_sinfo = dgetrs(A, A_piv, Ytmp_glob_1[:,i2])

    for i1 in range(e1-s1+1):  
        Ytmp_loc[:] =  Ytmp_glob_1[s1+i1, 0:e2+1-s2]
        subcomm_2.Allgatherv(Ytmp_loc, [Ytmp_glob_2, size_1, disp_1])
        X_glob_2[i1,:] = Ytmp_glob_2[:]
        X_glob_2[i1,:], B_sinfo = dgetrs(B, B_piv, X_glob_2[i1,:])

    X[p1:e1-s1+p1+1,p2:e2-s2+p2+1] = X_glob_2[:, s2:e2+1]

    return X

kron_solve_par_pyccel = epyccel(kron_solve_par_pyccel, header, libs = ['lapack'],compiler = 'mpif90')

header ='#$ header function kron_solve_par_bnd_pyccel(double[:,:](order=F), int, int, double[:,:](order=F), int, int, double[:,:](order=F), double[:,:](order=F), int[:], int[:], int[:], int[:], int[:], int[:], int[:], int[:], int[:])'
def kron_solve_par_bnd_pyccel(A_bnd, la, ua, B_bnd, lb, ub, X, Y, points, pads, starts, ends, subcoms, size_0, disp_0, size_1, disp_1):
    
    from mpi4py import MPI
    from scipy.linalg.lapack import dgbtrf, dgbtrs
    from numpy import zeros, zeros_like
    from numpy import shape
    ierr = -1
   
    
    subcomm_1 = subcoms[0]
    subcomm_2 = subcoms[1]

    s1 = starts[0]
    s2 = starts[1]
    e1 = ends[0]
    e2 = ends[1]
    n1 = points[0]
    n2 = points[1]
    p1 = pads[0]
    p2 = pads[1]
  
    Y_glob_1 = zeros((n1))
    Ytmp_glob_1 = zeros((n1, e2-s2+1),dtype = 'double',order = 'F')
    Ytmp_glob_2 = zeros((n2), dtype = 'double',order = 'F')
    X_glob_2 = zeros((e1-s1+1, n2),dtype = 'double',order = 'F')
   
    B_finfo = -1
    A_finfo = -1
    A_sinfo = -1
    B_sinfo = -1
    A_piv   = zeros(min(n1,n2),dtype = 'int',order = 'F')
    B_piv   = zeros(min(n1,n2),dtype = 'int',order = 'F')
    sh = shape(A_bnd)
    
    A_bnd, A_piv, A_finfo = dgbtrf(A_bnd, la, ua)
    B_bnd, B_piv, B_finfo = dgbtrf(B_bnd, lb, ub)
    Y_loc = zeros(e1-s1+1)
    Ytmp_loc = zeros(e2-s2+1)

    for i2 in range(e2-s2+1):
        Y_loc[:] = Y[p1:e1-s1+p1+1, i2+p2]
        subcomm_1.Allgatherv(Y_loc, [Y_glob_1, size_0, disp_0])
        Ytmp_glob_1[:,i2] = Y_glob_1[:] 
        Ytmp_glob_1[:,i2], A_sinfo = dgbtrs(A_bnd, la, ua,Ytmp_glob_1[:,i2], A_piv)

    for i1 in range(e1-s1+1):  
        Ytmp_loc[:] =  Ytmp_glob_1[s1+i1, 0:e2+1-s2]
        subcomm_2.Allgatherv(Ytmp_loc, [Ytmp_glob_2, size_1, disp_1])
        X_glob_2[i1,:] = Ytmp_glob_2[:]
        X_glob_2[i1,:], B_sinfo = dgbtrs(B_bnd, lb, ub, X_glob_2[i1,:], B_piv)

    X[p1:e1-s1+p1+1,p2:e2-s2+p2+1] = X_glob_2[:, s2:e2+1]

    return X

kron_solve_par_bnd_pyccel = epyccel(kron_solve_par_bnd_pyccel, header, libs = ['lapack'],compiler = 'mpif90')


