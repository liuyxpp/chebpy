# -*- coding: utf-8 -*-
"""
chebpy.cheba
============

Applications of Chebyshev spectral methods to solve PDEs.

"""

import numpy as np
from math import cos, acos

from chebpy import cheb_fast_transform, cheb_inverse_fast_transform

__all__ = ['cheb_mde_splitting_pseudospectral',
          ]

def cheb_mde_splitting_pseudospectral(W, L, Ns):
    '''
    Solve a modified diffusion equation (MDE) via Strang operator splitting
    as time-stepping scheme and pseudospectral methods on Chebyshev grids.

    The MDE is:
        dq/dt = Dq + Wq
    where D is Laplace operator.
    '''

    ds = 1. / (Ns -1)
    N = np.size(W) - 1
    u = np.ones(N+1)
    u[0] = 0.; u[N] = 0.
    k2 = (np.pi * np.pi) / (L * L) * np.arange(N+1) * np.arange(N+1)
    expw = np.exp(-0.5 * ds * W)
    for i in xrange(Ns-1):
        u = expw * u
        ak = cheb_fast_transform(u) * np.exp(-ds * k2)
        u = cheb_inverse_fast_transform(ak)
        u = expw * u
        u[0] = 0.; u[N] = 0.

    return u

    

