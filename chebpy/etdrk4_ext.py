# -*- coding: utf-8 -*-
"""
chebpy.etdrk4_ext
=================

Extended ETDRK4 classes.

"""

import numpy as np
from scipy.fftpack import fft, ifft, fft2, ifft2

from chebpy import BC, DIRICHLET
from chebpy import cheb_D2_mat_dirichlet_dirichlet, cheb_D2_mat_robin_robin
from chebpy import cheb_D2_mat_dirichlet_robin, cheb_D2_mat_robin_dirichlet
from chebpy import etdrk4_coeff_nondiag_krogstad
from chebpy import etdrk4_coeff_scale_square_krogstad
from chebpy import etdrk4_coeff_contour_hyperbolic_krogstad

__all__ = ['ETDRK4FxCy2',  # ETDRK4 Fourier x and Chebyshev y
           ]


class ETDRK4FxCy2(object):
    def __init__(self, Lx, Ly, Nx, Ny, Ns, h=None, c=1.0,
                 lbc=BC(), rbc=BC(), algo=1, scheme=1):
        '''
        The PDE is in 2D,
            du/dt = cLu - wu
        where u=u(x,y), L=d^2/dx^2 + d^2/dy^2, w=w(x,y), c is a constant.
        First, perform a FFT in x direction to obtain
            du(kx,y)/dt = c L u(kx,y) - {kx^2 u(kx,y) + Fx[w(x,y)u(x,y)]}
        where L = D^2, with D^2 the Chebyshev 2nd order differential matrix,
        and kx^2 the d^2/dx^2 in Fourier space, see detail in
        the Notebook (page 2014.5.5).

        The defaut left BC and right BC are DBCs.

        Test: PASSED, 2014.5.8. (However, very small ds is required!)

        :param:Lx: physical size of the 1D spacial grid.
        :param:Lx: physical size of the 1D spacial grid.
        :param:Ns: number of grid points in time.
        :param:lbc: left boundary condition.
        :type:lbc: class BC
        :param:rbc: right boundary condition.
        :type:rbc: class BC
        :param:h: time step.
        :param:algo: algorithm for calculation of RK4 coefficients.
        :param:scheme: RK4 scheme.
        '''
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.Ns = Ns
        if h is None:
            self.h = 1. / (Ns - 1)
        else:
            self.h = h
        self.c = c
        self.lbc = lbc
        self.rbc = rbc
        self.algo = algo
        self.scheme = scheme

        self.update()

    def update(self):
        Nx, Lx = self.Nx, self.Lx
        kx = np.arange(Nx)
        kxm = kx - Nx
        kx[Nx/2+1:] = kxm[Nx/2+1:]
        kx[Nx/2] = 0.0
        self.k2 = ((2 * np.pi / Lx) * kx)**2
        L = self._calc_operator()  # d^2/dx^2 operator on CGL grid.
        self._calc_RK4_coeff(L)

    def _calc_operator(self):
        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                D1, L, y = cheb_D2_mat_dirichlet_dirichlet(self.Ny)
            else:
                D1, L, y = cheb_D2_mat_dirichlet_robin(self.Ny,
                                                       self.rbc.beta)
        else:
            if self.rbc.kind == DIRICHLET:
                D1, L, y = cheb_D2_mat_robin_dirichlet(self.Ny,
                                                       self.lbc.beta)
            else:
                D1, L, y = cheb_D2_mat_robin_robin(self.Ny,
                                                   self.lbc.beta,
                                                   self.rbc.beta)

        self.y = .5 * (y + 1) * self.Ly
        L = (4. / self.Ly**2) * L  # map [0, Ly] onto [-1, 1]
        return L

    def _calc_RK4_coeff(self, L):
        L = self.c * L  # the actual operator
        h = self.h
        c = 1.0
        M = 32  # for both circular and hyperbolic contours
        R = 15.  # for circular contour only
        if self.scheme == 1:
            if self.algo == 0:
                self.E, self.E2, self.f1, self.f2, self.f3, \
                    self.f4, self.f5, self. f6 = \
                    etdrk4_coeff_nondiag_krogstad(L, h, M, R)
            elif self.algo == 1:
                self.E, self.E2, self.f1, self.f2, self.f3, \
                    self.f4, self.f5, self.f6 = \
                    etdrk4_coeff_contour_hyperbolic_krogstad(L, h, c, M)
            elif self.algo == 2:
                self.E, self.E2, self.f1, self.f2, self.f3, \
                    self.f4, self.f5, self.f6 = \
                    etdrk4_coeff_scale_square_krogstad(L, h)
            else:
                raise ValueError('No such ETDRK4 coefficient algorithm!')
            self.Q = None
        else:
            raise ValueError('No such ETDRK4 scheme!')

    def solve(self, w, u0, q=None):
        '''
            w = w(x,y), u0 = q(x,y,t=0), q = q(x,y,t)
        '''
        u = u0.copy()
        Ns = self.Ns
        k2 = self.k2
        E, E2 = self.E, self.E2
        f1, f2, f3 = self.f1, self.f2, self.f3
        f4, f5, f6 = self.f4, self.f5, self.f6
        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                v, W = u[:, 1:-1], -w[:, 1:-1]
            else:
                v, W = u[:, :-1], -w[:, :-1]
        else:
            if self.rbc.kind == DIRICHLET:
                v, W = u[:, 1:], -w[:, 1:]
            else:
                v, W = u, -w

        if self.scheme == 1:
            v_out, v_all = etdrk4fxcy_scheme_krogstad2(Ns, W, v, k2, E, E2,
                                                       f1, f2, f3, f4, f5, f6)
        else:
            raise ValueError('No such ETDRK4 scheme!')

        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                u[:, 1:-1] = v_out
                if q is not None:
                    q[:, :, 1:-1] = v_all
            else:
                u[:, :-1] = v_out
                if q is not None:
                    q[:, :, :-1] = v_all
        else:
            if self.rbc.kind == DIRICHLET:
                u[:, 1:] = v_out
                if q is not None:
                    q[:, :, 1:] = v_all
            else:
                u = v_out
                if q is not None:
                    q[1:, :, :] = v_all

        return u


def etdrk4fxcy_scheme_krogstad2(Ns, w, v, k2,
                                E, E2, f1, f2, f3, f4, f5, f6):
    '''
    w = w(x,y)
    v = v(x,y)
    k2: the Laplacian in x dimension in Fourier space.
    The size of E, E2, f1, f2, f3, f4, f5, f6 is (Ny, Ny)
    FFT in x and Chebyshev in y.
    Krogstad ETDRK4, whose stiff order is 3 better than Cox-Matthews ETDRK4.

    Test: PASSED, 2014.5.8. (should use very small ds!)
    '''
    vk = fft(v, axis=0)
    ak = np.zeros_like(vk)  # should be vk not v because of complex numbers
    bk = np.zeros_like(vk)
    ck = np.zeros_like(vk)
    Nx, Ny = v.shape
    q = np.zeros((Ns-1, Nx, Ny))

    for s in xrange(Ns-1):
        vk = fft(v, axis=0)
        Nu = w * v
        Nuk = fft(Nu, axis=0)
        for i in xrange(Nx):
            Nuk[i] = Nuk[i] - k2[i]*vk[i]
            ak[i] = np.dot(E2, vk[i]) + np.dot(f1, Nuk[i])
        a = ifft(ak, axis=0).real
        Na = w * a
        Nak = fft(Na, axis=0)
        for i in xrange(Nx):
            Nak[i] = Nak[i] - k2[i]*ak[i]
            bk[i] = ak[i] + np.dot(f2, Nak[i]-Nuk[i])
        b = ifft(bk, axis=0).real
        Nb = w * b
        Nbk = fft(Nb, axis=0)
        for i in xrange(Nx):
            Nbk[i] = Nbk[i] - k2[i]*bk[i]
            ck[i] = np.dot(E, vk[i]) \
                + np.dot(f3, Nuk[i]) \
                + np.dot(f4, Nbk[i]-Nuk[i])
        c = ifft(ck, axis=0).real
        Nc = w * c
        Nck = fft(Nc, axis=0)
        for i in xrange(Nx):
            Nck[i] = Nck[i] - k2[i]*ck[i]
            vk[i] = ck[i] + np.dot(f4, Nak[i]) \
                + np.dot(f5, Nuk[i]+Nck[i]) \
                + np.dot(f6, Nak[i]+Nbk[i])
        v = ifft(vk, axis=0).real
        q[s] = v[:, :]

    return v, q


def etdrk4fxycz_scheme_krogstad(Ns, w, v, k2,
                                E, E2, f1, f2, f3, f4, f5, f6):
    '''
    w = w(x,y,z)
    v = v(x,y,z)
    k2: the Laplacian in x and y dimension in Fourier space.
    The size of E, E2, f1, f2, f3, f4, f5, f6 is (Nz, Nz)
    FFT in x and y, Chebyshev in z.
    Krogstad ETDRK4, whose stiff order is 3 better than Cox-Matthews ETDRK4.

    Test: None.
    '''
    vk = fft2(v, axes=(0, 1))
    ak = np.zeros_like(vk)
    bk = np.zeros_like(vk)
    ck = np.zeros_like(vk)
    Nx, Ny, Nz = v.shape
    q = np.zeros((Ns-1, Nx, Ny, Nz))

    for s in xrange(Ns-1):
        vk = fft2(v, axes=(0, 1))
        Nu = w * v
        Nuk = fft2(Nu, axes=(0, 1))
        for i in xrange(Nx):
            for j in xrange(Ny):
                Nuk[i, j] = Nuk[i, j] - k2[i, j] * vk[i, j]
                ak[i, j] = np.dot(E2[i, j], vk[i, j]) \
                    + np.dot(f1[i, j], Nuk[i, j])
        a = ifft2(ak, axes=(0, 1)).real
        Na = w * a
        Nak = fft2(Na, axes=(0, 1))
        for i in xrange(Nx):
            for j in xrange(Ny):
                Nak[i, j] = Nak[i, j] - k2[i, j] * ak[i, j]
                bk[i, j] = ak[i, j] \
                    + np.dot(f2[i, j], Nak[i, j]-Nuk[i, j])
        b = ifft2(bk, axes=(0, 1)).real
        Nb = w * b
        Nbk = fft2(Nb, axes=(0, 1))
        for i in xrange(Nx):
            for j in xrange(Ny):
                Nbk[i, j] = Nbk[i, j] - k2[i, j] * bk[i, j]
                ck[i, j] = np.dot(E[i, j], vk[i, j]) \
                    + np.dot(f3[i, j], Nuk[i, j]) \
                    + np.dot(f4[i, j], Nbk[i, j]-Nuk[i, j])
        c = ifft2(ck, axes=(0, 1)).real
        Nc = w * c
        Nck = fft2(Nc, axes=(0, 1))
        for i in xrange(Nx):
            for j in xrange(Ny):
                Nck[i, j] = Nck[i, j] - k2[i, j] * ck[i, j]
                vk[i, j] = ck[i, j] + np.dot(f4[i, j], Nak[i, j]) \
                    + np.dot(f5[i, j], Nuk[i, j]+Nck[i, j]) \
                    + np.dot(f6[i, j], Nak[i, j]+Nbk[i, j])
        v = ifft2(vk, axes=(0, 1)).real
        q[s] = v[:, :, :]

    return v
