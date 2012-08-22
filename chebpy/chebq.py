# -*- coding: utf-8 -*-
"""
chebpy.chebq
============

Chebyshev quadrature.

"""

import numpy as np
from numpy.fft import ifft

__all__ = ['cheb_quadrature_cgl',
           'clencurt_weights',
           'clencurt_weights_fft',
           'cheb_quadrature_clencurt',
          ]

def cheb_quadrature_cgl(f):
    '''
    Chebyshev-Gauss 2nd kind quadrature.
    \integrate[-1,+1] sqrt(1-x^2) g(x) dx ~ \sum_i{1}{N-1} w_i * g(x_i)
    on
        x_i = cos(i*pi/N), i = 0, 1, 2, ..., N
    with
        w_i = (pi/N) * (1 - x_i^2), i = 1, 2, ..., N-1

    Note: from numeric experiments, this method is much less accurate than
    Clenshaw-Curtis quadrature.
    '''

    N = np.size(f) - 1
    theta = np.arange(N+1) * np.pi / N
    x = np.cos(theta)
    #w = np.zeros_like(x)
    #w += np.pi / N
    w = np.pi/N * (1-x**2)
    #w[0] *= .5
    #w[N] *= .5
    return np.dot(w[1:N], f[1:N] / np.sqrt(1-x[1:N]**2))


def clencurt_weights(N):
    '''
    The clenshaw-Curtis quadrature weights are:
        w_n = (4/(r_n*N)) * \sum_{k=0,even}{N} (1/r_k * cos(pi*k*n/N) / 
                (1 - k^2), n = 0, 1, 2, ..., N
    Thus, w_0 = w_N, and w_0 = 1 / (N^2 - 1) for even N, w_0 = 1 / N^2
    for odd N.

    Ref. Trefethen LN, 2000, p128.
    '''

    theta = np.arange(N+1) * np.pi / N
    x = np.cos(theta)
    w = np.zeros_like(x)
    ii = np.arange(1,N)
    v = np.ones(N-1)
    if N % 2 == 0:
        w[0] = 1. / (N**2 - 1)
        w[N] = w[0]
        for k in xrange(1,N/2):
            v -= 2. * np.cos(2*k*theta[ii]) / (4*k**2 - 1.)
        v -= np.cos(N*theta[ii]) / (N**2 - 1.)
    else:
        w[0] = 1. / N**2
        w[N] = w[0]
        for k in xrange(1,(N-1)/2+1):
            v -= 2. * np.cos(2*k*theta[ii]) / (4*k**2 - 1.)
    w[ii] = 2 * v / N
    return w


def clencurt_weights_fft(N):
    '''
    Compute the (N+1) nodes and weights for Clenshaw-Curtis quadrature
    on the interval [-1, 1]. Using FFT algorithm, the weights and nodes are
    computed in linear time.
    '''

    c = np.zeros(N+1)
    if N % 2 == 0:
        c[0:N+2:2] = 2. / (1. - np.arange(0,N+2,2)**2) 
    else:
        c[0:N:2] = 2. / (1. - np.arange(0,N,2)**2) 
    c = np.concatenate((c, np.flipud(c[1:N])))
    
    v = np.real(ifft(c))

    w = 2. * v[0:N+1]
    w[0] *= .5
    w[N] *= .5

    return w


def cheb_quadrature_clencurt(f):
    '''
    Integrate f in the range [-1, 1] by Clenshaw-Curtis quadrature.
    '''
    N = np.size(f) - 1
    w = clencurt_weights_fft(N)
    return np.dot(w, f)


