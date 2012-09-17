# -*- coding: utf-8 -*-
"""
chebpy.integral
===============

Numerical integration on equispaced grid.

"""

import numpy as np
from scipy.linalg import inv
from scipy.fftpack import dst

__all__ = ['complex_contour_integral',
           'etdrk4_coeff_nondiag', # complex contour integration
           'phi_contour_hyperbolic',
           'etdrk4_coeff_contour_hyperbolic',
           'etdrk4_coeff_scale_square', # scale and square
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
    

def etdrk4_coeff_nondiag(L, h, M=32, R=1.0):
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


def phi_contour_hyperbolic(L, h, l=0, M=32):
    '''
    Evaluate phi_l(h*L) using complex contour integral methods with hyperbolic contour.

    phi_l(z) = [phi_{l-1}(z) - phi_{l-1}(0)] / z, with
    phi_0(z) = exp(z)
    For example:
        phi_1(z) = [exp(z) - 1] / z
        phi_2(z) = [exp(z) - z - 1] / z^2
        phi_3(z) = [exp(z) - z^2/2 - z - 1] / z^3
    '''

    N, N = L.shape
    I = np.eye(N)
    phi = 1j * np.zeros((N,N))

    theta = np.pi * (2. * np.arange(M+1) / M - 1.)
    u = 1.0818 * theta / np.pi
    mu = 0.5 * 4.4921 * M
    alpha = 1.1721

    s = mu * (1 - np.sin(alpha - u*1j))
    v = np.cos(alpha - u*1j)

    if l == 0:
        c = np.exp(h*s) * v
    else:
        c = np.exp(h*s) * v / (h*s)**l 

    for k in np.arange(M+1):
        sIA = inv(s[k] * I - L)
        phi += c[k] * sIA

    return np.real((0.5 * 4.4921 * 1.0818 / np.pi) * phi)


def etdrk4_coeff_contour_hyperbolic(L, h, M=32):
    '''
    Evaluate etdrk4 coefficients by complex contour integral using
    hyperbolic contour.
    The hyperbolic contour is suitable for evaluating the cofficients for
    diffusive PDEs, whose eigenvalues lie close to the negative real line.

    Ref:
        * Schmelzer, T.; Trefethen, L. N. **Evaluating Matrix Functions for
        Exponential Integrators Via Caratheodory-Fejer Approximation and Contour Integrals* 2007.
        * Weideman, J. A.; Trefethen, L. N. "Parabolic and Hyperbolic Contours for Computing the Bromwich Integral" Math. Comput. 2007, 76, 1341.
        * Trefethen, L. N.; Weideman, J. A. C.; Schmelzer, T.; "Talbot Quadratures and Rational Approximations" BIT Numer. Math. 2006, 46, 653. 
    '''

    E1 = phi_contour_hyperbolic(L, h, 0, M) # phi_0(h*L) = exp(h*L)
    E2 = phi_contour_hyperbolic(L, h/2, 0, M) # phi_0(h/2*L)
    Q = 0.5 * phi_contour_hyperbolic(L, h/2, 1, M)
    phi1 = phi_contour_hyperbolic(L, h, 1, M)
    phi2 = phi_contour_hyperbolic(L, h, 2, M)
    phi3 = phi_contour_hyperbolic(L, h, 3, M)
    f1 = phi1 - 3 * phi2 + 4 * phi3
    f2 = 2 * (phi2 - 2 * phi3)
    f3 = 4 * phi3 - phi2
    
    return E1, E2, Q, f1, f2, f3


def etdrk4_coeff_scale_square(z, k, d=7):
    '''
    Evaluate etdrk4 coefficients by scaling and squaring methods. 

    Ref:
        Berland, H.; Skaflestad, B.; Wright, W. M.; 
        **EXPINT - A Matlab Package for Exponential Integrators**
        ACM Math. Soft. 2007, 33, Article 4.
    '''
    pass


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

