# -*- coding: utf-8 -*-
"""
chebpy.tridiag
==============

Special algorithms for quasi-tridiagonal system

"""

import numpy as np
from scipy.linalg import inv

__all__ = ['solve_tridiag_thual',
           'solve_tridiag_complex_thual',
          ]

def solve_tridiag_thual(p, q, r, c, f):
    '''
    Solve quasi-tridiagonal system with dense top row:
        p_i u_{i-1} + q_i u_i + r_i u_{i+1} = f_i, i = 1, ..., N-1
        p_N u_{N-1} + q_N u_N = f_N
        c_0 u_0 + c_1 u_1 + ... + c_N u_N = f_0

    The algorithm is adopt from Peyret **Spectral Methods for Incompressible
    Viscous Flow** 2000, Appendix B.

    :param:p: 1D array of size N
    :param:q: 1D array of size N
    :param:r: 1D array of size N, only r_0, r_1, ..., r_{N-2} is used.
    :param:c: 1D array of size N+1
    :param:f: 1D array of size N+1
    '''

    p = 1.*p; q = 1.*q; r = 1.*r; c = 1.*c; f = 1.*f

    u = np.zeros_like(f)

    X = np.zeros_like(p)
    Y = np.zeros_like(p)
    N = p.size
    X[-1] = - p[-1] / q[-1]
    Y[-1] = f[-1] / q[-1]
    for i in np.arange(N-2, -1, -1):
        t = q[i] + r[i] * X[i+1]
        X[i] = -p[i] / t
        Y[i] = (f[i+1] - r[i] * Y[i+1]) / t

    theta = np.zeros_like(f)
    theta[0] = 1.
    lamb = np.zeros_like(f)
    lamb[0] = 0.
    for i in np.arange(1,N+1):
        theta[i] = X[i-1] * theta[i-1]
        lamb[i] = X[i-1] * lamb[i-1] + Y[i-1]

    Theta = np.sum(c * theta)
    Lamb = np.sum(c * lamb)
    u[0] = (f[0] - Lamb) / Theta

    for i in np.arange(N):
        u[i+1] = X[i] * u[i] + Y[i]

    return u


def solve_tridiag_complex_thual(p, q, r, c, f):
    '''
    Solve quasi-tridiagonal system with dense top row:
        p_i u_{i-1} + q_i u_i + r_i u_{i+1} = f_i, i = 1, ..., N-1
        p_N u_{N-1} + q_N u_N = f_N
        c_0 u_0 + c_1 u_1 + ... + c_N u_N = f_0

    The algorithm is adopt from Peyret **Spectral Methods for Incompressible
    Viscous Flow** 2000, Appendix B.

    :param:p: 1D array of size N
    :param:q: 1D array of size N
    :param:r: 1D array of size N, only r_0, r_1, ..., r_{N-2} is used.
    :param:c: 1D array of size N+1
    :param:f: 1D array of size N+1
    '''

    p = 1.*p; q = 1.*q; r = 1.*r; c = 1.*c; f = 1.*f

    u = np.zeros_like(f).astype(complex)

    X = np.zeros_like(p)
    Y = np.zeros_like(p)
    N = p.size
    X[-1] = - p[-1] / q[-1]
    Y[-1] = f[-1] / q[-1]
    for i in np.arange(N-2, -1, -1):
        t = q[i] + r[i] * X[i+1]
        X[i] = -p[i] / t
        Y[i] = (f[i+1] - r[i] * Y[i+1]) / t

    theta = np.zeros_like(f).astype(complex)
    theta[0] = 1.
    lamb = np.zeros_like(f).astype(complex)
    lamb[0] = 0.
    for i in np.arange(1,N+1):
        theta[i] = X[i-1] * theta[i-1]
        lamb[i] = X[i-1] * lamb[i-1] + Y[i-1]

    Theta = np.sum(c * theta)
    Lamb = np.sum(c * lamb)
    u[0] = (f[0] - Lamb) / Theta

    for i in np.arange(N):
        u[i+1] = X[i] * u[i] + Y[i]

    return u

