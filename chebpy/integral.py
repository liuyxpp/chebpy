# -*- coding: utf-8 -*-
"""
chebpy.integral
===============

Numerical integration on equispaced grid.

"""

import numpy as np
from scipy.linalg import expm, expm2, expm3, inv
from scipy.fftpack import dst
from scipy.io import loadmat, savemat

__all__ = ['complex_contour_integral',
           'oss_integral_direct',
           'oss_integral_weights',
           'oss_weights',
          ]

def oss_weights(N):
    '''
    This calculates the weights for OSS integrals.
    The weights can be used for any OSS integration with N+1 nodes,
    where f_0 and f_N must be 0.
    See oss_integral_weights for more details.
    '''
    m = np.arange(1, N)
    c = 2 / np.pi / m
    c[1::2] = 0.
    w = dst(c, type=1) / N

    return w


def oss_integral_weights(f, w=None):
    '''
    Integrate f in the range (0, 1) obtained via OSS method by weights.
    The grid is
    x_0 x_1 x_2 ... x_N

    :param:f: f_0, f_1, ..., f_N, only f_1, f_2, ..., f_{N-1} is used.
    '''
    N = np.size(f) - 1
    if w is None:
        w = oss_weights(N)
    return np.dot(w, f[1:N])


def oss_integral_direct(f):
    '''
    Integrate f in the range (0, 1) obtained via OSS method directly.
    The grid is
    x_0 x_1 x_2 ... x_N

    :param:f: f_0, f_1, ..., f_N, only f_1, f_2, ..., f_{N-1} is used.
    '''
    N = np.size(f) - 1
    ak = dst(f[1:N], type=1) / N
    k = np.arange(1,N,2)
    return 2/np.pi * np.sum(ak[0::2]/k)
    

def complex_contour_integral(f, z, M=32, R=1.0):
    '''
    Evaluate the complex contour integral of the form
        (1/(2*pi*i)) \int_G f(t)/(t-z)dt
    '''
    theta = np.linspace(.5/M, 1-.5/M, M) * np.pi
    r = R * np.exp(1j * theta)

    In = 0.
    for j in xrange(M):
        t = r[j]
        In += f(t) * t / (t - z)

    return np.real(In) / M

