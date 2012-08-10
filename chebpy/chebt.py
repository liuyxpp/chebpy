# -*- coding: utf-8 -*-
"""
chebpy.chebt
============

Chebyshev transform.

"""

import numpy as np
from numpy.fft import fft

__all__ = ['cheb_fast_transform',
           'cheb_inverse_fast_transform']

def cheb_fast_transform(f, full=False):
    '''
    Fast forward Chebyshev transform via FFT.
    The size of f is N+1, and the size of ff is 2N.
        FFT(ff) = 2*DCT(f)
    while the forward Chebyshev transform is
        {2*DCT(f)[0] + 2*DCT(f)[N])/2N + 2*DCT(f)/N 

    :param:f: data to be transformed.
    :param:full: return the transform with redundant part.
    '''

    # f{0, 1, 2, ..., N}
    N = np.size(f) - 1
    if N == 0:
        return 0
    # Construct even function of f
    # ff = f{0, 1, 2, ..., N-1, N, N-1, N-2, ..., 2, 1}
    #    = f{0, 1, 2, ..., N-1, N} : f{N-1, N-2, ..., 2, 1}
    # For NumPy, f[1:N] = f{1, 2, ..., N-2, N-1}
    ff = np.concatenate((f, np.flipud(f[1:N])))
    # Perform FFT and retain the real part
    FF = np.real(fft(ff))

    FF[0] = .5 * FF[0]
    FF[N] = .5 * FF[N]
    if full:
        return FF / N
    else:
        # Only first N+1 terms are useful
        return FF[0:N+1] / N


def cheb_inverse_fast_transform(f, full=False):
    '''
    Fast backwark Chebyshev transform via FFT.
    The size of f is N+1, and the size of ff is 2N.
        FFT(ff) = 2*DCT(f)
    while the backward Chebyshev transform is
        \sum_j{0}{N} f_k cos(j*k*pi/N)
        = (2*f_0 + 2*(-1)^N*f_N) + \sum_j{1}{N-1} f_k cos(j*k*pi/N) 
        = DCT(f')
    where 
        f' = {2*f[0], f[1], f[2], ..., f[N-1], 2*f[N]}

    :param:f: data to be transformed.
    :param:full: return the transform with redundant part.
    '''

    # f{0, 1, 2, ..., N}
    N = np.size(f) - 1
    if N == 0:
        return 0
    # Construct even function of f
    # ff = f{0, 1, 2, ..., N-1, N, N-1, N-2, ..., 2, 1}
    #    = f{0, 1, 2, ..., N-1, N} : f{N-1, N-2, ..., 2, 1}
    # For NumPy, f[1:N] = f{1, 2, ..., N-2, N-1}
    f[0] = 2. * f[0]
    f[N] = 2. * f[N]
    ff = np.concatenate((f, np.flipud(f[1:N])))
    # Perform FFT and retain the real part
    FF = np.real(fft(ff))

    if full:
        return .5 * FF
    else:
        # Only first N+1 terms are useful
        return .5 * FF[0:N+1]


