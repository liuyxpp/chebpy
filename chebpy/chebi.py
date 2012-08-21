# -*- coding: utf-8 -*-
"""
chebpy.chebi
============

Chebyshev interpolations.

"""

import numpy as np
from chebpy import almost_equal

__all__ = ['barycentric_weights',
           'barycentric_weights_cgl',
           'barycentric_weights_cg',
           'barycentric_matrix',
           'cheb_barycentric_matrix',
           'interpolation_point',
           'interpolation_1d',
           'interpolation_2d',
           'cheb_interpolation_point',
           'cheb_interpolation_1d',
           'cheb_interpolation_2d',
          ]

def barycentric_weights(x):
    '''
    Compute the Barycentric weights for Barycentric Lagrange interpolation.

    See 
    * Berrut JP and Trefethn LN, SIAM Review, 2004, 45, 501-517
    * Kopriva DA, Implementing Spectral Methods for Partial Differential 
    Equations: Algorithms for Scientists and Engineers, 2009, Springer
    '''
    N = np.size(x) - 1
    w = np.ones_like(x)
    for j in xrange(1, N+1):
        for k in xrange(j):
            w[k] *= (x[k] - x[j])
            w[j] *= (x[j] - x[k])
    return 1. / w


def barycentric_weights_cg(N):
    '''
    Barycentric weights for Chebyshev Gaussian grids.
    The CG grids are
        x_j = cos[(2j+1)*pi/(2N+2)], j = 0, 1, 2, ..., N
    The weights are
        w_j = (-1)^j * sin[(2j+1)*pi/(2N+2)]

    '''
    ii = np.arange(N+1)
    return np.power(-1,ii) * np.sin((2*ii+1)*np.pi/(2*N+2))
    

def barycentric_weights_cgl(N):
    '''
    Barycentric weights for Chebyshev Gaussian grids.
    The CG grids are
        x_j = cos(j*pi/N), j = 0, 1, 2, ..., N
    The weights are
        w_j = (-1)^d_j, with
        d_j = 1/2, j = 0 or j = N
        d_j = 1,   j = 1, 2, ..., N-1

    '''
    ii = np.arange(N+1)
    w = np.ones(N+1) * np.power(-1,ii)
    w[0] *= .5; w[N] *= .5
    return w


def barycentric_matrix(y, x, w):
    '''
    Computation of matrix T_kj for interpolation between two sets of points.

    :param:y: set of points to be interpolated.
    :param:x: set of points containing source data.
    :param:w: Barycentric weights

    '''

    M = np.size(y) - 1
    N = np.size(x) - 1
    T = np.zeros((M+1,N+1))
    for k in xrange(M+1):
        row_has_match = False
        for j in xrange(N+1):
            if almost_equal(y[k], x[j]):
                row_has_match = True
                T[k,j] = 1.
                break
        if not row_has_match:
            t = w / (y[k] - x)
            T[k,:] = t / np.sum(t)
    return T


def cheb_barycentric_matrix(y, N):
    ii = np.arange(N+1)
    x = np.cos(ii * np.pi / N)
    w = barycentric_weights_cgl(N)
    return barycentric_matrix(y, x, w)


def interpolation_point(y, f, x, w):
    '''
    Barycentric Lagrange Interpolation for a single point of the most
    general form.

    :param:x: the point where the function value to be evaluated.
    :param:f: the function value at the specified grids.
    :param:xx: the locations array.
    :param:w: the Barycentric weights.
    '''

    N = np.size(f) - 1
    for j in xrange(N+1):
        if almost_equal(y, x[j]):
            return f[j]
    t = w / (y - x)
    return np.sum(t * f) / np.sum(t)


def cheb_interpolation_point(y, f):
    '''
    Barycentric Lagrange Interpolation for a single point of the
    Chebyshev Gauss-Lobatto form.

    :param:x: the point where the function value to be evaluated.
    :param:f: the function value at the specified grids.
    '''

    N = np.size(f) - 1
    ii = np.arange(N+1)
    x = np.cos(ii * np.pi / N)
    w = barycentric_weights_cgl(N)
    return interpolation_point(y, f, x, w)


def interpolation_1d(y, f, x, w):
    T = barycentric_matrix(y, x, w)
    return np.dot(T, f)


def cheb_interpolation_1d(y, f):
    N = np.size(f) - 1
    T = cheb_barycentric_matrix(y, N)
    return np.dot(T, f)


def interpolation_2d(y1, y2, f, x1, x2, w1, w2):
    '''
    Interpolate from Nx x Ny to Mx x My.

    :param:f: f[Ny, Nx]
    '''

    # y dimension T2 = My x Ny
    T2 = barycentric_matrix(y2, x2, w2)
    # F2 = My x Ny .dot. Ny x Nx = My x Nx
    F2 = np.dot(T2, f)
    # x dimension T1 = Mx x Nx
    T1 = barycentric_matrix(y1, x1, w1)
    # f_out = My x Mx = My x Nx .dot. Nx x Mx = F2 .dot. T1.T
    return np.dot(F2, T1.T)


def cheb_interpolation_2d(y1, y2, f):
    Ny, Nx = np.array(f.shape) - 1
    T2 = cheb_barycentric_matrix(y2, Ny)
    F2 = np.dot(T2, f)
    T1 = cheb_barycentric_matrix(y1, Nx)
    print T2.shape, f.shape, F2.shape, T1.shape
    return np.dot(F2, T1.T)

