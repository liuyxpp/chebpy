# -*- coding: utf-8 -*-
"""
chebpy.etdrk4
=============

Numerical integration on equispaced grid.

"""

import numpy as np
from scipy.linalg import expm, expm2, expm3, inv
from scipy.fftpack import dst
from scipy.io import loadmat, savemat

__all__ = ['etdrk4_coeff_nondiag', # complex contour integration
           'phi_contour_hyperbolic',
           'etdrk4_coeff_contour_hyperbolic',
           'etdrk4_coeff_scale_square', # scale and square
           'etdrk4_scheme_coxmatthews',
           'etdrk4_scheme_krogstad',
           'etdrk4_coeff_nondiag_krogstad',
           'etdrk4_coeff_contour_hyperbolic_krogstad',
           'etdrk4_coeff_scale_square_krogstad',
          ]

def etdrk4_coeff_nondiag(L, h, M=32, R=1.0):
    '''
    Evaluate the coefficients Q, f1, f2, f3 of ETDRK4 for
    non-diagonal case.

    '''

    A = h * L
    N, N = L.shape
    I = np.eye(N)

    E = expm(A)
    E2 = expm(A/2)
    
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
    f2 = 2 * (h/M) * np.real(f2)
    f3 = (h/M) * np.real(f3)
    Q = (h/M) * np.real(Q)
    
    return (E, E2, Q, f1, f2, f3)


def etdrk4_coeff_nondiag_krogstad(L, h, M=32, R=1.0):
    pass


def phi_contour_hyperbolic(z, l=0, M=32):
    '''
    Evaluate phi_l(h*L) using complex contour integral methods with hyperbolic contour.

    phi_l(z) = [phi_{l-1}(z) - phi_{l-1}(0)] / z, with
    phi_0(z) = exp(z)
    For example:
        phi_1(z) = [exp(z) - 1] / z
        phi_2(z) = [exp(z) - z - 1] / z^2
        phi_3(z) = [exp(z) - z^2/2 - z - 1] / z^3
    '''

    N, N = z.shape
    I = np.eye(N)
    phi = 1j * np.zeros((N,N))

    #theta = np.pi * (2. * np.arange(M+1) / M - 1.)
    theta = np.pi * ((2. * np.arange(M) + 1) / M - 1.)
    u = 1.0818 * theta / np.pi
    mu = 0.5 * 4.4921 * M
    alpha = 1.1721

    s = mu * (1 - np.sin(alpha - u*1j))
    v = np.cos(alpha - u*1j)

    if l == 0:
        c = np.exp(s) * v
    else:
        c = np.exp(s) * v / (s)**l 

    for k in np.arange(M):
        sIA = inv(s[k] * I - z)
        phi += c[k] * sIA

    return np.real((0.5 * 4.4921 * 1.0818 / np.pi) * phi)


def etdrk4_coeff_contour_hyperbolic(L, h, M=32):
    '''
    Evaluate etdrk4 coefficients by complex contour integral using
    hyperbolic contour.
    The hyperbolic contour is suitable for evaluating the cofficients for
    diffusive PDEs, whose eigenvalues lie close to the negative real line.

    Practice:
        This seems less accurate than cicular contour plus expm(L*h) and
        expm(L*h/2).
        M = 32 is optimized.

    Ref:
        * Schmelzer, T.; Trefethen, L. N. **Evaluating Matrix Functions for
        Exponential Integrators Via Caratheodory-Fejer Approximation and Contour Integrals* 2007.
        * Weideman, J. A.; Trefethen, L. N. "Parabolic and Hyperbolic Contours for Computing the Bromwich Integral" Math. Comput. 2007, 76, 1341.
        * Trefethen, L. N.; Weideman, J. A. C.; Schmelzer, T.; "Talbot Quadratures and Rational Approximations" BIT Numer. Math. 2006, 46, 653. 
    '''

    #E1 = phi_contour_hyperbolic(L*h, 0, M) # phi_0(h*L) = exp(h*L)
    E1 = expm(h*L)
    #E2 = phi_contour_hyperbolic(L*h/2, 0, M) # phi_0(h/2*L)
    E2 = expm(h/2*L)
    Q = h * 0.5 * phi_contour_hyperbolic(L*h/2, 1, M)
    phi1 = phi_contour_hyperbolic(L*h, 1, M)
    phi2 = phi_contour_hyperbolic(L*h, 2, M)
    phi3 = phi_contour_hyperbolic(L*h, 3, M)
    f1 = h* (phi1 - 3 * phi2 + 4 * phi3)
    f2 = h * 2 * (phi2 - 2 * phi3)
    f3 = h* (4 * phi3 - phi2)
    
    return E1, E2, Q, f1, f2, f3


def etdrk4_coeff_contour_hyperbolic_krogstad(L, h, M=32):
    '''
    Evaluate etdrk4 coefficients by complex contour integral using
    hyperbolic contour for Krogstad scheme.
    '''

    #E1 = phi_contour_hyperbolic(L*h, 0, M) # phi_0(h*L) = exp(h*L)
    E1 = expm(h*L)
    #E2 = phi_contour_hyperbolic(L*h/2, 0, M) # phi_0(h/2*L)
    E2 = expm(h/2*L)
    f1 = h * 0.5 * phi_contour_hyperbolic(L*h/2, 1, M)
    f2 = h * phi_contour_hyperbolic(L*h/2, 2, M)
    phi1 = phi_contour_hyperbolic(L*h, 1, M)
    phi2 = phi_contour_hyperbolic(L*h, 2, M)
    phi3 = phi_contour_hyperbolic(L*h, 3, M)
    f3 = h * phi1
    f4 = h * 2 * phi2
    f5 = h * (4 * phi3 - phi2)
    f6 = - h * 4 * phi3
    
    return E1, E2, f1, f2, f3, f4, f5, f6


def etdrk4_coeff_scale_square(L, h, d=7):
    '''
    Evaluate etdrk4 coefficients by scaling and squaring methods. 

    Ref:
        Berland, H.; Skaflestad, B.; Wright, W. M.; 
        **EXPINT - A Matlab Package for Exponential Integrators**
        ACM Math. Soft. 2007, 33, Article 4.
    '''
    Ns = int(1/h)
    data_name = 'benchmark/scale_square_data/etdrk4_phi_N256_Ns' + str(Ns-1) + '.mat'
    phimat = loadmat(data_name)
    E1 = phimat['ez']
    E2 = phimat['ez2']
    Q = h * 0.5 * phimat['phi_12']
    phi1 = phimat['phi_1']
    phi2 = phimat['phi_2']
    phi3 = phimat['phi_3']
    f1 = h * (phi1 - 3 * phi2 + 4 * phi3)
    f2 = h * 2 * (phi2 - 2 * phi3)
    f3 = h * (4 * phi3 - phi2)
    
    return E1, E2, Q, f1, f2, f3


def etdrk4_coeff_scale_square_krogstad(L, h, d=7):
    pass


def etdrk4_scheme_coxmatthews(Ns, w, v, E, E2, Q, f1, f2, f3):
    '''
    Krogstad ETDRK4, whose stiff order is 3 better than Cox-Matthews ETDRK4.
    '''
    for j in xrange(Ns-1):
        Nu = w * v
        a = np.dot(E2, v) + np.dot(Q, Nu)
        Na = w * a
        b = np.dot(E2, v) + np.dot(Q, Na)
        Nb = w * b
        c = np.dot(E2, a) + np.dot(Q, 2*Nb-Nu)
        Nc = w * c
        v = np.dot(E, v) + np.dot(f1, Nu) + np.dot(f2, Na+Nb) + \
            np.dot(f3, Nc)

    return v


def etdrk4_scheme_krogstad(Ns, w, v, E, E2, f1, f2, f3, f4, f5, f6):
    '''
    Krogstad ETDRK4, whose stiff order is 3 better than Cox-Matthews ETDRK4.
    '''

    for j in xrange(Ns-1):
        Nu = w * v
        a = np.dot(E2, v) + np.dot(f1, Nu)
        Na = w * a
        b = a + np.dot(f2, Na-Nu)
        Nb = w * b
        c = np.dot(E, v) + np.dot(f3, Nu) + np.dot(f4, Nb-Nu)
        Nc = w * c
        v = c + np.dot(f4, Na) + np.dot(f5, Nu+Nc) + np.dot(f6, Na+Nb)

    return v
