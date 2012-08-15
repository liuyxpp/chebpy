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


def cheb_D1_fft(v):
    '''
    Evaluate 1st derivative with Chebyshev polynomials via fast Fourier 
    transform (FFT).
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

