# -*- coding: utf-8 -*-
"""
chebpy.chebd
============

Chebyshev differentiation.

Speed comparison:
For large N, (typically N > 100)
    cheb_D1_dct > cheb_D1_fft > cheb_D1_fchebt >> cheb_D1_mat
For small N,
    cheb_D1_mat > cheb_D1_fchebt > cheb_D1_dct > cheb_D1_fft
This is because the recursion in computing derivative coefficients
is expensive for large N.

"""

import numpy as np
from numpy.fft import fft, ifft
from scipy.fftpack import dct, idct

from chebpy import cheb_fast_transform, cheb_inverse_fast_transform

__all__ = ['cheb_D1_mat',
           'cheb_D2_mat_dirichlet_dirichlet',
           'cheb_D2_mat_dirichlet_robin',
           'cheb_D2_mat_robin_dirichlet',
           'cheb_D2_mat_robin_robin',
           'cheb_D2_mat_robin_robin_1',
           'cheb_D1_fft',
           'cheb_D1_dct',
           'cheb_D1_fchebt']

def cheb_D1_mat(N):
    '''
    Evaluate 1st derivative using matrix multiply approach.
    The off-diagonal entries of the differentiation matrix D are:
        D_ij = (c_i / c_j) * (-1)^(i+j) / (x_i - x_j)
    or in matrix form:
        C_ij = [(-1)^i * c_i] * [(-1)^j / c_j]
        dX_ij = x_i - x_j
    The diagonal entries are:
        D_ii = - \sum_i{0}{N}{i<>j} D_ij

    :param:N: dimension of the matrix
    :return: the defferentiation matrix and the Chebyshev grids

    '''

    if N == 0:
        return (0., 1.)

    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    x.shape = (N+1, 1)

    c = np.ones( (N+1, 1) )
    c[0] = 2.
    c[N] = 2.
    c *= np.power(-1, ii).reshape( (N+1, 1) )

    X = np.tile(x, (1, N+1))
    dX = X - X.T

    # D = (c * (1./c)') ./ (dX + eye(N+1)) in Matlab convention
    D = np.dot(c, 1.0/c.T) / (dX + np.eye(N+1))
    D -= np.diag(np.sum(D, axis=1))
    return (D, x)


def cheb_D2_mat_dirichlet_dirichlet(N):
    '''
    Chebyshev differentiation matrix subjecting to DBC-DBC.
    DBC-DBC: 
        Dirichlet at x=-1 and x=+1

    Ref:
        Weideman, J. A.; Reddy, S. C. "A Matlab Differentiation Matrix Suite" ACM Trans. Math. Softw. 2000, 26, 465.

    :res:D1t: 1st order differentiation matrix, size: N x N
    :res:D2t: 2nd order differentiation matrix, size: N x N
    :res:x: Chebyshev points = cos(i/N*pi), i = 0, 1, ..., N
    '''
    D0 = np.eye(N+1)
    D1, x = cheb_D1_mat(N) # Note: x is a column vector
    D2 = np.dot(D1, D1)

    J = np.arange(1,N)
    K = np.arange(1,N)
    X, Y = np.meshgrid(K, J)
    X = X.T; Y = Y.T
    D1t = D1[X,Y]
    D2t = D2[X,Y]

    return D1t, D2t, x


def cheb_D2_mat_dirichlet_robin(N, kb):
    '''
    Chebyshev differentiation matrix subjecting to DBC-RBC.
    DBC-RBC: 
        Dirichlet at x=-1 and Robin at x=+1
    Note: 
        DBC-NBC is a special case with kb = 0

    Ref:
        Weideman, J. A.; Reddy, S. C. "A Matlab Differentiation Matrix Suite" ACM Trans. Math. Softw. 2000, 26, 465.

    :res:D1t: 1st order differentiation matrix, size: N x N
    :res:D2t: 2nd order differentiation matrix, size: N x N
    :res:x: Chebyshev points = cos(i/N*pi), i = 0, 1, ..., N
    '''
    D0 = np.eye(N+1)
    D1, x = cheb_D1_mat(N) # Note: x is a column vector
    D2 = np.dot(D1, D1)

    J = np.arange(1,N)
    K = np.arange(0,N)
    xjrow = 1 - x[J].T
    xkcol = 1 - x[K]
    oner = np.ones(xkcol.size)
    oner.shape = (oner.size, 1) # to column vector

    fac0 = np.dot(oner, 1/xjrow)
    fac1 = np.dot(xkcol, 1/xjrow)
    X, Y = np.meshgrid(K, J)
    X = X.T; Y = Y.T
    D1t = fac1 * D1[X,Y] - fac0 * D0[X,Y]
    D2t = fac1 * D2[X,Y] - 2 * fac0 * D1[X,Y]

    cfac = D1[0,0] + kb;
    fcol1 = -cfac * D0[:N,0] + (1 + cfac * xkcol.T) * D1[:N,0]
    fcol2 = -2 * cfac * D1[:N,0] + (1 + cfac * xkcol.T) * D2[:N,0]
    fcol1.shape = (fcol1.size, 1)
    fcol2.shape = (fcol2.size, 1)
    D1t = np.hstack((fcol1, D1t))
    D2t = np.hstack((fcol2, D2t))

    return D1t, D2t, x


def cheb_D2_mat_robin_dirichlet(N, ka):
    '''
    Chebyshev differentiation matrix subjecting to RBC-DBC.
    RBC-DBC: 
        Robin at x=-1 and Dirichlet at x=+1
    Note: 
        NBC-DBC is a special case with ka = 0

    Ref:
        Weideman, J. A.; Reddy, S. C. "A Matlab Differentiation Matrix Suite" ACM Trans. Math. Softw. 2000, 26, 465.

    :res:D1t: 1st order differentiation matrix, size: N x N
    :res:D2t: 2nd order differentiation matrix, size: N x N
    :res:x: Chebyshev points = cos(i/N*pi), i = 0, 1, ..., N
    '''
    D0 = np.eye(N+1)
    D1, x = cheb_D1_mat(N) # Note: x is a column vector
    D2 = np.dot(D1, D1)

    J = np.arange(1,N)
    K = np.arange(1,N+1)
    xjrow = 1 + x[J].T
    xkcol = 1 + x[K]
    oner = np.ones(xkcol.size)
    oner.shape = (oner.size, 1) # to column vector

    fac0 = np.dot(oner, 1/xjrow)
    fac1 = np.dot(xkcol, 1/xjrow)
    X, Y = np.meshgrid(K, J)
    X = X.T; Y = Y.T
    D1t = fac1 * D1[X,Y] + fac0 * D0[X,Y]
    D2t = fac1 * D2[X,Y] + 2 * fac0 * D1[X,Y]

    cfac = D1[-1,-1] + ka;
    lcol1 = -cfac * D0[1:,-1] + (1 - cfac * xkcol.T) * D1[1:,-1]
    lcol2 = -2 * cfac * D1[1:,-1] + (1 - cfac * xkcol.T) * D2[1:,-1]
    lcol1.shape = (lcol1.size, 1)
    lcol2.shape = (lcol2.size, 1)
    D1t = np.hstack((D1t, lcol1))
    D2t = np.hstack((D2t, lcol2))

    return D1t, D2t, x


def cheb_D2_mat_robin_robin(N, ka, kb):
    '''
    Chebyshev differentiation matrix subjecting to RBC-RBC.
    RBC-RBC: Robin at x=-1 and x=+1
    Note: 
        NBC-RBC is a special case with ka = 0
        RBC-NBC is a special case with kb = 0
        NBC-NBC is a special case with ka = 0 and kb = 0

    Ref:
        Weideman, J. A.; Reddy, S. C. "A Matlab Differentiation Matrix Suite" ACM Trans. Math. Softw. 2000, 26, 465.

    :res:D1t: 1st order differentiation matrix, size: (N+1) x (N+1)
    :res:D2t: 2nd order differentiation matrix, size: (N+1) x (N+1)
    :res:x: Chebyshev points = cos(i/N*pi), i = 0, 1, ..., N
    '''
    D0 = np.eye(N+1)
    D1, x = cheb_D1_mat(N) # Note: x is a column vector
    D2 = np.dot(D1, D1)

    J = np.arange(1,N)
    K = np.arange(0,N+1)
    xjrow = 1 / (1 - x[J].T**2)
    xkcol0 = 1 - x[K]**2
    xkcol1 = -2 * x[K]
    xkcol2 = -2 * np.ones_like(xkcol0)

    fac0 = np.dot(xkcol0, xjrow)
    fac1 = np.dot(xkcol1, xjrow)
    fac2 = np.dot(xkcol2, xjrow)

    X, Y = np.meshgrid(K, J)
    X = X.T; Y = Y.T
    D1t = fac0 * D1[X,Y] + fac1 * D0[X,Y]
    D2t = fac0 * D2[X,Y] + 2 * fac1 * D1[X,Y] + fac2 * D0[X,Y]

    omx = 0.5 * (1 - x[K])
    opx = 0.5 * (1 + x[K])

    r0 = opx + (0.5 + D1[0,0] + kb) * xkcol0 / 2
    r1 = 0.5 - (0.5 + D1[0,0] + kb) * x
    r2 = -0.5 - D1[0,0] - kb
    rcol1 = r0.T * D1[:,0] + r1.T * D0[:,0]
    rcol2 = r0.T * D2[:,0] + 2 * r1.T * D1[:,0] + r2.T * D0[:,0]
    rcol1.shape = (rcol1.size, 1)
    rcol2.shape = (rcol2.size, 1)

    l0 = omx + (0.5 - D1[-1,-1] - ka) * xkcol0 / 2
    l1 = -0.5 + (D1[-1,-1] + ka  -0.5) *x
    l2 = D1[-1,-1] + ka - 0.5
    lcol1 = l0.T * D1[:,-1] + l1.T * D0[:,-1]
    lcol2 = l0.T * D2[:,-1] + 2 * l1.T * D1[:,-1] + l2.T * D0[:,-1]
    lcol1.shape = (lcol1.size, 1)
    lcol2.shape = (lcol2.size, 1)

    D1t = np.hstack((rcol1, D1t, lcol1))
    D2t = np.hstack((rcol2, D2t, lcol2))

    return D1t, D2t, x


def cheb_D2_mat_robin_robin_1(N, ka, kb):
    '''
    Chebyshev differentiation matrix subjecting to RBC-RBC.
    RBC-RBC: Robin at x=-1 and x=+1
    Note: 
        NBC-RBC is a special case with ka = 0
        RBC-NBC is a special case with kb = 0
        NBC-NBC is a special case with ka = 0 and kb = 0

    Ref:
        Weideman, J. A.; Reddy, S. C. "A Matlab Differentiation Matrix Suite" ACM Trans. Math. Softw. 2000, 26, 465.

    :res:D1t: 1st order differentiation matrix, size: (N+1) x (N+1)
    :res:D2t: 2nd order differentiation matrix, size: (N+1) x (N+1)
    :res:x: Chebyshev points = cos(i/N*pi), i = 0, 1, ..., N
    '''
    D, x = cheb_D1_mat(N)
    D1 = np.zeros_like(D)
    D1[1:N,:] = D[1:N,:]
    L = np.dot(D, D1) 
    L[:,0] -= D[:,0] * kb
    L[:,N] -= D[:,N] * ka

    return L, x


def cheb_D1_fft(v):
    '''
    Evaluate 1st derivative with Chebyshev polynomials via fast Fourier 
    transform (FFT).

    Ref:
        Trefethen, LN **Spectral Methods in Matlab**, 2000

    :param:v: 1D numpy array contains the data to be differentiated.

    '''

    N = np.size(v) - 1
    if N == 0:
        return 0
    # Construct Chebyshev Gauss-Lobatto Grids
    ii = np.arange(N+1)
    x = np.cos(ii * np.pi / N) 

    # Projecting Gauss-Lobatto grids to equal-spaced grids 
    # theta = ii * pi / N
    V = np.concatenate((v, np.flipud(v[1:N])))
    U = np.real(fft(V))

    # 1st Derivative in spetral space for theta
    ii = np.arange(N)
    kk = np.concatenate((ii, [0], np.flipud(-ii[1:])))

    # Transfrom back to spatial space for theta
    W = np.real(ifft(1j * kk * U))

    # Projecting theta to x
    w = np.zeros(N+1)
    w[1:N] = -W[1:N] / np.sqrt(1. - x[1:N] * x[1:N])
    w[0] = np.sum(ii * ii * U[ii]) / N + .5 * N * U[N]
    w[N] = np.sum(np.power(-1, ii+1) * ii * ii * U[ii]) / N + \
            .5 * np.power(-1, N+1) * N * U[N]

    return w


def cheb_D1_coefficients(fk):
    N = np.size(fk) - 1
    dfk = np.zeros(N+1)
    dfk[N] = 0
    dfk[N-1] = 2. * N * fk[N]
    for k in xrange(N-2, 0, -1):
        dfk[k] = 2. * (k+1) * fk[k+1] + dfk[k+2]
    dfk[0] = fk[1] + .5 * dfk[2]
    return dfk


def cheb_D1_fchebt(v):
    '''
    Evaluate 1st derivative with Chebyshev polynomials via fast Fourier 
    transform (FFT).

    Ref:
        Kopriva, DA **Implementing Spectral Methods for Partial Differential
        Equations**, 2009

    :param:v: 1D numpy array contains the data to be differentiated.

    '''

    N = np.size(v) - 1
    if N == 0:
        return 0

    # Perfrom forward Chebyshev transform to obtain the coefficients
    V = cheb_fast_transform(v)

    # Compute the 1st Derivative coefficients in spetral space
    W = cheb_D1_coefficients(V)

    # Transfrom back to spatial space for theta
    w = cheb_inverse_fast_transform(W)

    return w


def cheb_D1_dct(v):
    '''
    Evaluate 1st derivative with Chebyshev polynomials via discrete cosine
    transform (DCT).
    :param:v: 1D numpy array contains the data to be differentiated.

    '''

    N = np.size(v) - 1
    if N == 0:
        return 0
    ii = np.arange(N+1)
    x = np.cos(ii * np.pi / N)
    ii = np.arange(N)
    u = dct(v, type=1)
    U = np.concatenate((u, np.flipud(u[1:N])))
    kk = np.concatenate((ii, [0], np.flipud(-ii[1:])))
    W = np.real(ifft(1j * kk * U))
    w = np.zeros(N+1)
    w[1:N] = -W[1:N] / np.sqrt(1. - x[1:N] * x[1:N])
    w[0] = np.sum(ii * ii * U[ii]) / N + .5 * N * U[N]
    w[N] = np.sum(np.power(-1, ii+1) * ii * ii * U[ii]) / N + \
            .5 * np.power(-1, N+1) * N * U[N]

    return w

