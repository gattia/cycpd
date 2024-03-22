# cython: infer_types=True
import numpy as np

cimport cython
from libc.math cimport exp as c_exp
from libc.math cimport log as c_log
from libc.math cimport pow as c_pow

# from libcpp.vector cimport vector

ctypedef fused my_type:
    double

cdef extern from "vfastexp.h":
    # fast implementation of exp
    double exp_approx "EXP" (double)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Do division using C?
def expectation(my_type[:,:] X, my_type[:,:] TY, my_type sigma2, int M, int N, int D, double w):
    # TY is MxD (same as Y - it is just transformed Y). M = rows of points Y, and D is dimensions.
    # tic = time.time()
    cdef Py_ssize_t x_i_shape, x_j_shape, y_i_shape, y_j_shape
    cdef Py_ssize_t i, j, k
    cdef double c, c_divisor

    x_i_shape = X.shape[0]
    x_j_shape = X.shape[1]
    y_i_shape = TY.shape[0]
    y_j_shape = TY.shape[1]

    cdef my_type tmp, tmp2, tmp_total, P_tmp, Pt1_sum, P1_sum, Np
    cdef my_type two_sigma2, neg_tmp_total, machine_error

    two_sigma2 = 2 * sigma2
    machine_error = np.finfo(float).eps
    c = (2 * np.pi * sigma2)
    c = c_pow(c, D/2)
    c_divisor = w / (1-w)
    c = c * c_divisor
    c = c * M / N

    if my_type is int:
        dtype = np.intc
    elif my_type is double:
        dtype = np.double
    elif my_type is cython.longlong:
        dtype = np.longlong


    # cdef my_type[:, :] P
    P = np.zeros((M, N), dtype=dtype)
    cdef my_type[:,:] P_view = P

    # cdef my_type[:] den
    den = np.zeros(x_i_shape, dtype=dtype)
    cdef my_type[:] den_view = den

    # cdef my_type[:] den_tmp
    den_tmp = np.zeros(x_i_shape, dtype=dtype)
    cdef my_type[:] den_tmp_view = den_tmp

    # cdef my_type[:] Pt1
    Pt1 = np.zeros(N, dtype=dtype)
    cdef my_type[:] Pt1_view = Pt1

    # cdef my_type[:] P1
    P1 = np.zeros(M, dtype=dtype)
    cdef my_type[:] P1_view = P1

    cdef my_type[:, :] X_view = X
    cdef my_type[:, :] TY_view = TY

    # toc = time.time()
    # print('Time to load the arrays was: {}'.format(toc-tic))
    # tic = time.time()

    for i in range(y_i_shape):
        # print('at j: {}'.format(j))
        for j in range(x_i_shape):
            tmp_total = 0
            for k in range(y_j_shape):
                tmp_total += c_pow(X[j,k] - TY[i,k], 2.)
            P_view[i, j] = c_exp(-tmp_total / two_sigma2)
            # P_view[i, j] = exp_approx(P_tmp)
            # P_view[i, j] = np.exp(P_tmp)

    # toc = time.time()
    # print('Time to do loop 1 was: {}'.format(toc-tic))
    # tic = time.time()

    for i in range(y_i_shape):
        for j in range(x_i_shape):
            den_view[j] = den_view[j] + P_view[i, j]
            # den_view[j] = den_tmp_view[j]
            # if den_tmp_view[j] == 0:
            #     den_view[j] = machine_error
            # else:

    for j in range(x_i_shape):
        den_view[j] += c
        if den_view[j] == 0:
            den_view[j] = machine_error


    # toc = time.time()
    # print('Time to do loop 2 was: {}'.format(toc-tic))
    # tic = time.time()

    Np = 0
    for i in range(y_i_shape):
        for j in range(x_i_shape):
            # P   = np.divide(P, den[None,:])
            P_view[i, j] = P_view[i, j] / den_view[j]
            # Pt1 = np.sum(P, axis=0)
            Pt1_view[j] +=  P_view[i, j]
            # P1  = np.sum(P, axis=1)
            P1_view[i] +=  P_view[i, j]
        # Np  = np.sum(P1)
        Np += P1_view[i]

    # toc = time.time()
    # print('Time to do loop 3 was: {}'.format(toc-tic))
    # tic = time.time()

    return P, Pt1, P1, Np

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Do division using C?
def expectation_2(my_type[:,:] X, my_type[:,:] TY, my_type sigma2, int M, int N, int D, double w):
    # TY is MxD (same as Y - it is just transformed Y). M = rows of points Y, and D is dimensions.
    # tic = time.time()
    # cdef Py_ssize_t x_i_shape, x_j_shape, y_i_shape, y_j_shape
    if my_type is double:
        dtype = np.double

    cdef Py_ssize_t n, m, d  # i, j, k

    cdef double ksig, diff, w_tmp, den, tmp_total, Np, E # den = sp in original matlab, tmp_total = razn
    # cdef double [:] temp_x
    temp_x = np.zeros(D, dtype=np.double)
    cdef double [:] temp_x_view = temp_x

    ksig = -2.0 * sigma2
    w_tmp = (w * M * c_pow(-ksig * 3.14159265358979,0.5*D))/((1-w)*N) # c in the original manuscript

    cdef my_type[:, :] X_view = X
    cdef my_type[:, :] TY_view = TY

     # Make P array
    # cdef my_type[:] P
    P = np.zeros(M, dtype=dtype)
    cdef my_type[:] P_view = P

    # Make P1 array
    # cdef my_type[:] P1
    P1= np.zeros(M, dtype=dtype)
    cdef my_type[:] P1_view = P1

    # Make Pt1 array
    # cdef my_type[:] Pt1
    Pt1= np.zeros(N, dtype=dtype)
    cdef my_type[:] Pt1_view = Pt1

    # Make Pt1 array
    # cdef my_type[:, :] Px
    Px= np.zeros((M, D), dtype=dtype)
    cdef my_type[:, :] Px_view = Px


    for n in range(N):
        den = 0
        for m in range(M):
            tmp_total = 0
            for d in range(D):
                diff = X_view[n, d] - TY_view[m, d]
                diff = diff * diff
                tmp_total += diff
            P_view[m] = c_exp(tmp_total/ksig)
            den += P_view[m]
        den += w_tmp
        Pt1_view[n] = 1-w_tmp/den

        for d in range(D):
            temp_x_view[d] = X_view[n, d] / den

        for m in range(M):
            P1_view[m] += P_view[m] / den

            for d in range(D):
                Px_view[m, d] += temp_x_view[d] * P_view[m]
        
        if den <= 0:
            den = np.finfo(float).eps
            
        E += -c_log(den)

    Np = 0
    for m in range(M):
        Np += P1_view[m]

    E += D * Np * c_log(sigma2)/2

    return P1, Pt1, Px, Np, E




@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def initialize_sigma2(my_type[:,:] X, my_type[:,:] Y):
    # For X and Y of shape 10,000x3 this code takes an
    # average of 0.253 s (253ms) compared to the original
    # code which takes 4.4 s

    cdef Py_ssize_t x_i_shape, x_j_shape, y_i_shape, y_j_shape
    cdef my_type diff, diff2, sigma2, size
    cdef my_type err = 0
    cdef Py_ssize_t i, j, k

    x_i_shape = X.shape[0]
    x_j_shape = X.shape[1]
    y_i_shape = Y.shape[0]
    y_j_shape = Y.shape[1]

    assert x_j_shape == y_j_shape

    for i in range(x_i_shape):
        for j in range(y_i_shape):
            for k in range(x_j_shape):
                diff = X[i, k] - Y[j, k]
                diff2 = c_pow(diff, 2.)
                err += diff2

    size = (x_i_shape * y_i_shape * x_j_shape)
    sigma2 = err / size

    return sigma2
