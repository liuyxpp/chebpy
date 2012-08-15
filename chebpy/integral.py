# -*- coding: utf-8 -*-
"""
chebpy.integral
===============

Numerical integration.

"""

import numpy as np
from scipy.linalg import inv

__all__ = ['complex_contour_integral',
           'etdrk4_coeff_ndiag',
          ]

def etdrk4_coeff_ndiag(L, h, M=32, R=1.0):
    '''
    Evaluate the coefficients Q, f1, f2, f3 of ETDRK4 for
    non-diagonal case.

    '''

    A = h * L
    N, N = L.shape
    I = np.eye(N)
    
    theta = np.linspace(.5/M, 1-.5/M, M) * np.pi
    r = R * np.exp(1j * theta)

    Z = 1j * np.zeros((N, N))
    f1 = Z.copy(); f2 = Z.copy(); f3 = Z.copy(); Q = Z.copy()

    for j in xrange(M):
        z = r[j]
        zIA = inv(z * I - A)
        zIAz2 = zIA / z**2
        Q += zIA * (np.exp(z/2) - 1)
        f1 += zIAz2 * (-4 - z + np.exp(z) * (4 - 3*z + z**2))
        f2 += zIAz2 * (2 + z + np.exp(z) * (z - 2))
        f3 += zIAz2 * (-4 - 3*z - z*z + np.exp(z) * (4 - z))
    f1 = (h/M) * np.real(f1)
    f2 = (h/M) * np.real(f2)
    f3 = (h/M) * np.real(f3)
    Q = (h/M) * np.real(Q)
    
    return (Q, f1, f2, f3)


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

