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

from chebpy import DIRICHLET, NEUMANN, ROBIN
from chebpy import cheb_D2_mat_dirichlet_dirichlet, cheb_D2_mat_robin_robin
from chebpy import cheb_D2_mat_dirichlet_robin, cheb_D2_mat_robin_dirichlet

__all__ = ['BC', # class for boundary condition
           'ETDRK4', # ETDRK4 class
           'etdrk4_coeff_nondiag', # complex contour integration
           'phi_contour_hyperbolic',
           'etdrk4_coeff_contour_hyperbolic',
           'etdrk4_coeff_scale_square', # scale and square
           'etdrk4_scheme_coxmatthews',
           'etdrk4_scheme_krogstad',
           'etdrk4_coeff_nondiag_krogstad',
           'etdrk4_coeff_contour_hyperbolic_krogstad',
           'etdrk4_coeff_scale_square_krogstad',
          ]

class BC(object):
    def __init__(self, kind=None, vc=None):
        '''
        The boundary condition can be generally written as:
            alpha * du/dx + beta * u = gamma
        It is convenient to specified BC by a 3-element vector:
            (alpha, beta, gamma)
        :param:kind: kind of boundary conditions.
        :param:vc: boundary condition specified by a vec with 3 elements.
        '''
        if vc is not None and len(vc) != 3:
            raise ValueError('The vector to specify boundary condtion'
                             'must has 3 elements!')
        if kind == ROBIN and vc is None:
            raise ValueError('Robin BC needs a coefficient vector.')
            
        self.kind = kind
        if kind is None:
            if vc is None:
                self.kind = DIRICHLET
            else:
                self.kind = ROBIN

        if self.kind == DIRICHLET:
            self.alpha = 0
            self.beta = 1.
            self.gamma = 0
        elif self.kind == NEUMANN:
            self.alpha = 1.
            self.beta = 0
            self.gamma = 0
        elif kind == ROBIN:
            self.alpha = vc[0]
            self.beta = vc[1]
            self.gamma = vc[2]
        else:
            raise ValueError('kind ' + str(kind) + ' is not supported.')


class ETDRK4(object):
    def __init__(self, Lx, N, Ns, h=None, 
                 lbc=BC(), rbc=BC(), algo=1, scheme=1):
        '''
        The defaut left BC and right BC are DBCs.

        :param:Lx: physical size of the 1D spacial grid.
        :param:Ns: number of grid points in time.
        :param:lbc: left boundary condition.
        :type:lbc: class BC
        :param:rbc: right boundary condition.
        :type:rbc: class BC
        :param:h: time step.
        :param:save_all: is save all solutions for each time step?
        :param:algo: algorithm for calculation of RK4 coefficients.
        :param:scheme: RK4 scheme.
        '''
        self.Lx = Lx
        self.N = N
        self.Ns = Ns
        self.lbc = lbc
        self.rbc = rbc
        if h is None:
            self.h = 1. / (Ns - 1)
        else:
            self.h = h
        self.algo = algo
        self.scheme = scheme
        
        self.update()

    def update(self):
        self._calc_operator()
        self._calc_RK4_coeff()

    def _calc_operator(self):
        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                D1, L, x = cheb_D2_mat_dirichlet_dirichlet(self.N)
            else:
                D1, L, x = cheb_D2_mat_dirichlet_robin(self.N, 
                                                       self.rbc.beta)
        else:
            if self.rbc.kind == DIRICHLET:
                D1, L, x = cheb_D2_mat_robin_dirichlet(self.N, 
                                                       self.lbc.beta)
            else:
                D1, L, x = cheb_D2_mat_robin_robin(self.N, 
                                                   self.lbc.beta,
                                                   self.rbc.beta)

        self.x = .5 * (x + 1) * self.Lx
        self.L = (4. / self.Lx**2) * L

    def _calc_RK4_coeff(self):
        M = 32; R = 15.;
        if self.scheme == 0:
            if self.algo == 0:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_nondiag(self.L, self.h, M, R)
            elif self.algo == 1:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_contour_hyperbolic(self.L, self.h, M)
            elif self.algo == 2:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_scale_square(self.L, self.h)
            else:
                raise ValueError('No such ETDRK4 coefficient algorithm!')
            f4 = None; f5 = None; f6 = None
        elif self.scheme == 1:
            if self.algo == 0:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_nondiag_krogstad(self.L, self.h, M, R)
            elif self.algo == 1:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_contour_hyperbolic_krogstad(self.L, self.h, M)
            elif self.algo == 2:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_scale_square_krogstad(self.L, self.h)
            else:
                raise ValueError('No such ETDRK4 coefficient algorithm!')
            Q = None
        else:
            raise ValueError('No such ETDRK4 scheme!')

        self.E = E
        self.E2 = E2
        self.Q = Q
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        self.f5 = f5
        self.f6 = f6

    def solve(self, w, u0, q=None):
        '''
            dq/dt = Dq + Wq = Dq - wq
        '''
        u = u0.copy()
        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                v = u[1:-1]
                W = -w[1:-1]
            else:
                v = u[:-1]
                W = -w[:-1]
        else:
            if self.rbc.kind == DIRICHLET:
                v = u[1:]
                W = -w[1:]
            else:
                v = u
                W = -w
        E = self.E
        E2 = self.E2
        Q = self.Q
        f1 = self.f1
        f2 = self.f2
        f3 = self.f3
        f4 = self.f4
        f5 = self.f5
        f6 = self.f6
        if self.scheme == 0:
            v = etdrk4_scheme_coxmatthews(
                    self.Ns, W, v, E, E2, Q, f1, f2, f3)
        else:
            v = etdrk4_scheme_krogstad(
                    self.Ns, W, v, E, E2, f1, f2, f3, f4, f5, f6, q)

        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                u[1:-1] = v
            else:
                u[:-1] = v
        else:
            if self.rbc.kind == DIRICHLET:
                u[1:] = v
            else:
                u = v

        return (u, self.x)


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


def etdrk4_scheme_krogstad(Ns, w, v, E, E2, f1, f2, f3, f4, f5, f6, q=None):
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
        if q is not None:
            q[j+1] = v[:,0]

    return v

