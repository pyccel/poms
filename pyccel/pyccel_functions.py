from pyccel import epyccel

header = '#$ header function  kron_dot_pyccel_2d(int[:], int[:], int[:], double[:,:], double[:,:], double[:,:], double[:,:], double[:,:])'
def kron_dot_pyccel_2d(starts, ends, pads, X, X_tmp, Y, A, B):
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

#kron_dot_pyccel_2d = epyccel(kron_dot_pyccel_2d, header)

header='#$ header function kron_solve_serial_pyccel_2d(double[:,:](order = F),double[:,:](order = F),double[:,:](order = F),double[:,:](order = F),int[:], int[:])'
def kron_solve_serial_pyccel_2d(A, B, X, Y, points, pads):
    
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

#kron_solve_serial_pyccel_2d = epyccel(kron_solve_serial_pyccel_2d, header, libs = ['lapack'])


header ='#$ header function kron_solve_par_pyccel_2d(double[:,:](order=F), double[:,:](order=F), double[:,:](order=F), double[:,:](order=F) ,int[:], int[:], int[:], int[:], int[:],int[:], int[:],int[:],int[:])'
def kron_solve_par_pyccel_2d(A, B, X, Y, points, pads, starts, ends, subcoms, size_0, disp_0, size_1, disp_1):
    
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
    Ytmp_glob_1 = zeros((n1, n2),dtype = 'double',order = 'F')
    Ytmp_glob_2 = zeros((n2), dtype = 'double',order = 'F')
    X_glob_2 = zeros((n1, n2),dtype = 'double',order = 'F')
   
    B_finfo = -1
    A_finfo = -1
    A_sinfo = -1
    B_sinfo = -1
    A_piv   = zeros(min(n1,n2),dtype = 'int',order = 'F')
    B_piv   = zeros(min(n1,n2),dtype = 'int',order = 'F')

    A, A_piv, A_finfo = dgetrf(A)
    B, B_piv, B_finfo = dgetrf(B)
    Y_loc = zeros(n1)
    Ytmp_loc = zeros(n2)

    for i2 in range(n2):
        Y_loc[:] = Y[p1:n1+p1, i2+p2]
        subcomm_1.Allgatherv(Y_loc, [Y_glob_1, size_0, disp_0])
        Ytmp_glob_1[:,i2] = Y_glob_1[:] 
        Ytmp_glob_1[:,i2], A_sinfo = dgetrs(A, A_piv, Ytmp_glob_1[:,i2])

    for i1 in range(n1):  
        Ytmp_loc[:] =  Ytmp_glob_1[s1+i1, 0:n2]
        subcomm_2.Allgatherv(Ytmp_loc, [Ytmp_glob_2, size_1, disp_1])
        X_glob_2[i1,:] = Ytmp_glob_2[:]
        X_glob_2[i1,:], B_sinfo = dgetrs(B, B_piv, X_glob_2[i1,:])

        X[p1+i1,p2:n2+p2] = X_glob_2[:, s2:e2+1]

    return X

#kron_solve_par_pyccel_2d = epyccel(kron_solve_par_pyccel_2d, header, libs = ['lapack'],mpi = True)

header ='#$ header function kron_solve_par_bnd_pyccel_2d(double[:,:](order=F), int, int, double[:,:](order=F), int, int, double[:,:](order=F), double[:,:](order=F), int[:], int[:], int[:], int[:], int[:], int[:], int[:], int[:], int[:])'
def kron_solve_par_bnd_pyccel_2d(A_bnd, la, ua, B_bnd, lb, ub, X, Y, points, pads, starts, ends, subcoms, size_0, disp_0, size_1, disp_1):
    
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
  
    
    
    
   
    B_finfo = -1
    A_finfo = -1
    A_sinfo = -1
    B_sinfo = -1
    A_piv   = zeros(n1, dtype = 'int')
    B_piv   = zeros(n2, dtype = 'int')
    
    A_bnd, A_piv, A_finfo = dgbtrf(A_bnd, la, ua)
    B_bnd, B_piv, B_finfo = dgbtrf(B_bnd, lb, ub)

    X_glob      = zeros(n1)
    Y_glob      = zeros(n2)
    X_loc       = zeros(n1)
    Y_loc       = zeros(n2)
    

    for i2 in range(n2):
        X_loc[:] = Y[p1:n1+p1,i2+p2]
        subcomm_1.Allgatherv(X_loc, [X_glob, size_0, disp_0, MPI.DOUBLE])
        X_glob, A_sinfo = dgbtrs(A_bnd, la, ua, X_glob, A_piv)
        X[p1:n1+p1,p2+i2] = X_glob[:]

    for i1 in range(n1):  
        Y_loc[:] =  X[i1+p1,p2:n2+p2]
        subcomm_2.Allgatherv(Y_loc, [Y_glob, size_1, disp_1, MPI.DOUBLE])
        Y_glob, B_sinfo = dgbtrs(B_bnd, lb, ub, Y_glob, B_piv)
        X[p1+i1,p2:n2+p2] = Y_glob[:]

    return X

#kron_solve_par_bnd_pyccel_2d = epyccel(kron_solve_par_bnd_pyccel_2d, header, libs = ['lapack'], mpi = True)


header ='#$ header function kron_solve_par_bnd_pyccel_3d(double[:,:](order=F), int, int, double[:,:](order=F), int, int, double[:,:](order=F), int, int, double[:,:,:](order=F), double[:,:,:](order=F), int[:], int[:], int[:], int[:], int[:], int[:], int[:], int[:], int[:],int[:],int[:])'
def kron_solve_par_bnd_pyccel_3d(A_bnd, la, ua, B_bnd, lb, ub, C_bnd, lc, uc ,X, Y, points, pads, starts, ends, subcoms,
                                 size_0, disp_0, size_1, disp_1, size_2, disp_2):
    
    from mpi4py import MPI
    from scipy.linalg.lapack import dgbtrf, dgbtrs
    from numpy import zeros, zeros_like
    from numpy import shape
    ierr = -1
   
    
    subcomm_1 = subcoms[0]
    subcomm_2 = subcoms[1]
    subcomm_2 = subcoms[2]

    s1 = starts[0]
    s2 = starts[1]
    s3 = starts[2]
    e1 = ends[0]
    e2 = ends[1]
    e3 = ends[2]
    n1 = points[0]
    n2 = points[1]
    n3 = points[2]
    p1 = pads[0]
    p2 = pads[1]
    p3 = pads[2]
  

   
    B_finfo = -1
    A_finfo = -1
    A_sinfo = -1
    B_sinfo = -1
    C_sinfo = -1
    C_finfo = -1

    A_piv   = zeros(n1, dtype = 'int')
    B_piv   = zeros(n2, dtype = 'int')
    C_piv   = zeros(n3, dtype = 'int')
    
    A_bnd, A_piv, A_finfo = dgbtrf(A_bnd, la, ua)
    B_bnd, B_piv, B_finfo = dgbtrf(B_bnd, lb, ub)
    C_bnd, C_piv, C_finfo = dgbtrf(C_bnd, lc, uc)

    X_glob      = zeros(n1)
    Y_glob      = zeros(n2)
    Z_glob      = zeros(n3)
    X_loc       = zeros(n1)
    Y_loc       = zeros(n2)
    Z_loc       = zeros(n3)
    
    for i3 in range(n3):
        for i2 in range(n2):
            X_loc[:] = Y[p1:n1+p1,i2+p2, i3+p3]
            subcomm_1.Allgatherv(X_loc, [X_glob, size_0, disp_0, MPI.DOUBLE])
            X_glob, A_sinfo = dgbtrs(A_bnd, la, ua, X_glob, A_piv)
            X[p1:n1+p1, p2+i2, p3+i3] = X_glob[:]

    for i3 in range(n3):
        for i1 in range(n1):  
            Y_loc[:] =  X[i1+p1, p2:n2+p2, i3+p3]
            subcomm_2.Allgatherv(Y_loc, [Y_glob, size_1, disp_1, MPI.DOUBLE])
            Y_glob, B_sinfo = dgbtrs(B_bnd, lb, ub, Y_glob, B_piv)
            X[p1+i1, p2:n2+p2, p3+i3] = Y_glob[:]

    for i2 in range(n2):
        for i1 in range(n1):  
            Z_loc[:] =  X[i1+p1, i2+p2, p3:p3+n3]
            subcomm_2.Allgatherv(Z_loc, [Z_glob, size_2, disp_2, MPI.DOUBLE])
            Z_glob, C_sinfo = dgbtrs(C_bnd, lc, uc, Z_glob, C_piv)
            X[p1+i1, p2+i2, p3:p3+n3] = Z_glob[:]



    return X

#kron_solve_par_bnd_pyccel_3d = epyccel(kron_solve_par_bnd_pyccel_3d, header, libs = ['lapack'],mpi = True)

