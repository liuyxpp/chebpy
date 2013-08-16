# -*- coding: utf-8 -*-
"""
chebpy.etdrk4
=============

ETDRK4 class and related methods.

"""

import numpy as np
from scipy.linalg import expm, expm2, expm3, inv
from scipy.fftpack import dst, fft, ifft, fft2, ifft2, fftn, ifftn
from scipy.io import loadmat, savemat

from chebpy import BC, DIRICHLET, NEUMANN, ROBIN
from chebpy import cheb_D2_mat_dirichlet_dirichlet, cheb_D2_mat_robin_robin
from chebpy import cheb_D2_mat_dirichlet_robin, cheb_D2_mat_robin_dirichlet

__all__ = ['ETDRK4', # ETDRK4 class
           'ETDRK4FxCy', # ETDRK4 Fourier x and Chebyshev y
           'ETDRK4FxyCz', # ETDRK4 Fourier x, y, and Chebyshev z
           'ETDRK4Polar', # ETDRK4 in polar coordinates, Fourier theta and Chebyshev 
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

class ETDRK4(object):
    def __init__(self, Lx, N, Ns, h=None, c=1.0,  
                 lbc=BC(), rbc=BC(), algo=1, scheme=1):
        '''
        The PDE is
            du/dt = cLu - wu
        Here c is an constant, L is a linear operator, w is a function.

        The defaut left BC and right BC are DBCs.

        Test: PASSED, 2012, 2013

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
        self.c = c
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
        self.L = (4. / self.Lx**2) * L # map [0, Lx] onto [-1, 1]

    def _calc_RK4_coeff(self):
        L = self.c * self.L # the actual operator
        h = self.h
        c = 1.0
        M = 32; R = 15.;
        if self.scheme == 0:
            if self.algo == 0:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_nondiag(L, h, M, R)
            elif self.algo == 1:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_contour_hyperbolic(L, h, M)
            elif self.algo == 2:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_scale_square(L, h)
            else:
                raise ValueError('No such ETDRK4 coefficient algorithm!')
            f4 = None; f5 = None; f6 = None
        elif self.scheme == 1:
            if self.algo == 0:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_nondiag_krogstad(L, h, M, R)
            elif self.algo == 1:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_contour_hyperbolic_krogstad(L, h, c, M)
            elif self.algo == 2:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_scale_square_krogstad(L, h)
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
        u = u0.copy(); u.shape = (u.size, 1)
        E = self.E; E2 = self.E2; Q = self.Q
        f1 = self.f1; f2 = self.f2; f3 = self.f3
        f4 = self.f4; f5 = self.f5; f6 = self.f6
        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                v = u[1:-1]
                W = -w[1:-1]; W.shape = (W.size, 1)
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                      Q, f1, f2, f3, q[:,1:-1])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6, q[:,1:-1])
                    else:
                        v = etdrk4_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[1:-1] = v
            else:
                v = u[:-1]
                W = -w[:-1]; W.shape = (W.size, 1)
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                      Q, f1, f2, f3, q[:,:-1])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6, q[:,:-1])
                    else:
                        v = etdrk4_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:-1] = v
        else:
            if self.rbc.kind == DIRICHLET:
                v = u[1:]
                W = -w[1:]; W.shape = (W.size, 1)
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                      Q, f1, f2, f3, q[:,1:])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6, q[:,1:])
                    else:
                        v = etdrk4_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[1:] = v
            else:
                v = u
                W = -w; W.shape = (W.size, 1)
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3, q)
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6, q)
                    else:
                        v = etdrk4_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u = v

        return (u, self.x)


class ETDRK4FxCy(object):
    def __init__(self, Lx, Ly, Nx, Ny, Ns, h=None, c=1.0,  
                 lbc=BC(), rbc=BC(), algo=1, scheme=1):
        '''
        The PDE is in 2D,
            du/dt = cLu - wu
        where u=u(x,y), L=d^2/dx^2 + d^2/dy^2, w=w(x,y), c is a constant.
        First, do a FFT in x direction to obtain
            du(kx,y)/dt = c L u(kx,y) - Fx[w(x,y)u(x,y)]
        where L = D^2 - kx^2, with D^2 the Chebyshev 2-nd order differential matrix,
        and kx^2 the d^2/dx^2 in Fourier space, see detail in
        the Notebook (page 2013.8.2).

        The defaut left BC and right BC are DBCs.

        Test: PASSED 2013.08.09.
        Note: Cox-Matthews scheme not tested.

        :param:Lx: physical size of the 1D spacial grid.
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
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
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
        Nx = self.Nx
        L = self._calc_operator() # the shape of coeff depends on BC
        N, N = L.shape
        I = np.eye(N)
        dim = [Nx, N, N]
        self.E = np.zeros(dim)
        self.E2 = np.zeros(dim)
        self.Q = np.zeros(dim)
        self.f1 = np.zeros(dim)
        self.f2 = np.zeros(dim)
        self.f3 = np.zeros(dim)
        self.f4 = np.zeros(dim)
        self.f5 = np.zeros(dim)
        self.f6 = np.zeros(dim)
        for i in xrange(Nx):
            if i < Nx/2+1:
                kx = i * (2 * np.pi / self.Lx)
            else:
                kx = (i - Nx) * (2 * np.pi / self.Lx)
            k2 = kx**2
            #L = self._calc_operator(k2)
            self._calc_RK4_coeff(i, L-k2*I)

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
        L = (4. / self.Ly**2) * L # map [0, Lx] onto [-1, 1]
        return L

    def _calc_RK4_coeff(self, i, L):
        L = self.c * L # the actual operator
        h = self.h
        c = 1.0
        M = 32; R = 15.;
        if self.scheme == 0:
            if self.algo == 0:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_nondiag(L, h, M, R)
            elif self.algo == 1:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_contour_hyperbolic(L, h, M)
            elif self.algo == 2:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_scale_square(L, h)
            else:
                raise ValueError('No such ETDRK4 coefficient algorithm!')
            self.E[i] = E[:,:]
            self.E2[i] = E2[:,:]
            self.Q[i] = Q[:,:]
            self.f1[i] = f1[:,:]
            self.f2[i] = f2[:,:]
            self.f3[i] = f3[:,:]
            f4 = None; f5 = None; f6 = None
        elif self.scheme == 1:
            if self.algo == 0:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_nondiag_krogstad(L, h, M, R)
            elif self.algo == 1:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_contour_hyperbolic_krogstad(L, h, c, M)
            elif self.algo == 2:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_scale_square_krogstad(L, h)
            else:
                raise ValueError('No such ETDRK4 coefficient algorithm!')
            self.E[i] = E[:,:]
            self.E2[i] = E2[:,:]
            Q = None
            self.f1[i] = f1[:,:]
            self.f2[i] = f2[:,:]
            self.f3[i] = f3[:,:]
            self.f4[i] = f4[:,:]
            self.f5[i] = f5[:,:]
            self.f6[i] = f6[:,:]
        else:
            raise ValueError('No such ETDRK4 scheme!')


    def solve(self, w, u0, q=None):
        '''
            w = w(x,y)
            u0 = q(x,y,t=0)
            q = q(x,y,t)
        '''
        u = u0.copy()
        E = self.E; E2 = self.E2; Q = self.Q
        f1 = self.f1; f2 = self.f2; f3 = self.f3
        f4 = self.f4; f5 = self.f5; f6 = self.f6
        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                v = u[:,1:-1]
                W = -w[:,1:-1]
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                Q, f1, f2, f3, q[:,:,1:-1])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                                   f4, f5, f6, q[:,:,1:-1])
                    else:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:,1:-1] = v
            else:
                v = u[:,:-1]
                W = -w[:,:-1]
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                Q, f1, f2, f3, q[:,:,:-1])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                                   f4, f5, f6, q[:,:,:-1])
                    else:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:,:-1] = v
        else:
            if self.rbc.kind == DIRICHLET:
                v = u[:,1:]
                W = -w[:,1:]
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                      Q, f1, f2, f3, q[:,:,1:])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                                   f4, f5, f6, q[:,:,1:])
                    else:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:,1:] = v
            else:
                v = u
                W = -w
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3, q)
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6, q)
                    else:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u = v

        return u


class ETDRK4FxyCz(object):
    def __init__(self, Lx, Ly, Lz, Nx, Ny, Nz, Ns, h=None, c=1.0,  
                 lbc=BC(), rbc=BC(), algo=1, scheme=1):
        '''
        The PDE is in 3D,
            du/dt = cLu - wu
        where u=u(t,x,y,z), L=d^2/dx^2 + d^2/dy^2, w=w(x,y,z), c is a constant.
        First, do a FFT in x and y direction to obtain
            du(kx,ky,z)/dt = c L u(kx,ky,z) - Fxy[w(x,y,z)u(t,x,y,z)]
        where L = D^2 - (kx^2 + ky^2), with D^2 the Chebyshev 2-nd order
        differential matrix with appropriate boundary conditions,
        and -kx^2 and -ky^2 are d^2/dx^2 and d^2/dy^2 in Fourier space, see
        detail in the Notebook (page 2013.8.2).

        The defaut left BC and right BC are DBCs.

        Test: None

        :param:Lx: physical size of the 1D spacial grid.
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
        self.Ly = Ly
        self.Lz = Lz
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
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
        Nx = self.Nx
        Ny = self.Ny
        L = self._calc_operator() # the shape of coeff depends on BC
        N, N = L.shape # N may be different than Nz+1 because of DBC 
        I = np.eye(N)
        dim = [Nx, Ny, N, N]
        self.E = np.zeros(dim)
        self.E2 = np.zeros(dim)
        self.Q = np.zeros(dim)
        self.f1 = np.zeros(dim)
        self.f2 = np.zeros(dim)
        self.f3 = np.zeros(dim)
        self.f4 = np.zeros(dim)
        self.f5 = np.zeros(dim)
        self.f6 = np.zeros(dim)
        for i in xrange(Nx):
            for j in xrange(Ny):
                if i < Nx/2+1:
                    kx = i * (2 * np.pi / self.Lx)
                else:
                    kx = (i - Nx) * (2 * np.pi / self.Lx)
                if j < Ny/2+1:
                    ky = j * (2 * np.pi / self.Ly)
                else:
                    ky = (j - Nx) * (2 * np.pi / self.Ly)
                k2 = kx**2 + ky**2
                #L = self._calc_operator(k2)
                self._calc_RK4_coeff(i, j, L-k2*I)

    def _calc_operator(self):
        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                D1, L, z = cheb_D2_mat_dirichlet_dirichlet(self.Nz)
            else:
                D1, L, z = cheb_D2_mat_dirichlet_robin(self.Nz, 
                                                       self.rbc.beta)
        else:
            if self.rbc.kind == DIRICHLET:
                D1, L, z = cheb_D2_mat_robin_dirichlet(self.Nz, 
                                                       self.lbc.beta)
            else:
                D1, L, z = cheb_D2_mat_robin_robin(self.Nz, 
                                                   self.lbc.beta,
                                                   self.rbc.beta)

        self.z = .5 * (z + 1) * self.Lz
        L = (4. / self.Lz**2) * L # map [0, Lz] onto [-1, 1]
        return L

    def _calc_RK4_coeff(self, i, j, L):
        L = self.c * L # the actual operator
        h = self.h
        c = 1.0
        M = 32; R = 15.;
        if self.scheme == 0:
            if self.algo == 0:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_nondiag(L, h, M, R)
            elif self.algo == 1:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_contour_hyperbolic(L, h, M)
            elif self.algo == 2:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_scale_square(L, h)
            else:
                raise ValueError('No such ETDRK4 coefficient algorithm!')
            self.E[i,j] = E[:,:]
            self.E2[i,j] = E2[:,:]
            self.Q[i,j] = Q[:,:]
            self.f1[i,j] = f1[:,:]
            self.f2[i,j] = f2[:,:]
            self.f3[i,j] = f3[:,:]
            f4 = None; f5 = None; f6 = None
        elif self.scheme == 1:
            if self.algo == 0:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_nondiag_krogstad(L, h, M, R)
            elif self.algo == 1:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_contour_hyperbolic_krogstad(L, h, c, M)
            elif self.algo == 2:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_scale_square_krogstad(L, h)
            else:
                raise ValueError('No such ETDRK4 coefficient algorithm!')
            self.E[i,j] = E[:,:]
            self.E2[i,j] = E2[:,:]
            Q = None
            self.f1[i,j] = f1[:,:]
            self.f2[i,j] = f2[:,:]
            self.f3[i,j] = f3[:,:]
            self.f4[i,j] = f4[:,:]
            self.f5[i,j] = f5[:,:]
            self.f6[i,j] = f6[:,:]
        else:
            raise ValueError('No such ETDRK4 scheme!')


    def solve(self, w, u0, q=None):
        '''
            w = w(x,y,z)
            u0 = q(t=0,x,y,z)
            q = q(t,x,y,z)
        '''
        u = u0.copy()
        E = self.E; E2 = self.E2; Q = self.Q
        f1 = self.f1; f2 = self.f2; f3 = self.f3
        f4 = self.f4; f5 = self.f5; f6 = self.f6
        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                v = u[:,:,1:-1]
                W = -w[:,:,1:-1]
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                      Q, f1, f2, f3, q[:,:,:,1:-1])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                                       f4, f5, f6, q[:,:,:,1:-1])
                    else:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:,:,1:-1] = v
            else:
                v = u[:,:,:-1]
                W = -w[:,:,:-1]
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                      Q, f1, f2, f3, q[:,:,:,:-1])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                                       f4, f5, f6, q[:,:,:,:-1])
                    else:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:,:,:-1] = v
        else:
            if self.rbc.kind == DIRICHLET:
                v = u[:,:,1:]
                W = -w[:,:,1:]
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                      Q, f1, f2, f3, q[:,:,:,1:])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                                    E, E2, f1, f2, f3, 
                                                    f4, f5, f6, q[:,:,:,1:])
                    else:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:,:,1:] = v
            else:
                v = u
                W = -w
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3, q)
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6, q)
                    else:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u = v

        return u


class ETDRK4Polar(object):
    def __init__(self, R, Nr, Nt, Ns, h=None, c=1.0,  
                 lbc=BC(), rbc=BC(), algo=1, scheme=1):
        '''
        The PDE is in the polar coordinate,
            du/dt = cLu - wu
        where u=u(r,theta), L=d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/dtheta^2,
        w=w(r,theta), c is a constant. Domain is
            theta [0, 2*pi]
            r     [0, R]
        First, do a FFT in theta axis to obtain
            du(r,kt)/dt = c L u(r,kt) - Ft[w(r,theta)u(r,theta)]
        where L = d^2/dr^2 + (1/r)d/dr - (1/r^2)kt^2*I
        See details in the Notebook (page 2013.8.15).

        The defaut left BC and right BC are RBCs.

        Test: PASSED 2013.8.15.

        :param:R: physical size of the disk.
        :param:Nr: r axis discretization, 0, 1, 2, ..., Nr. Nr must be ODD.
        :param:Nt: theta axis discretization, 0, 1, 2, ..., Nt-1
        :param:Ns: number of grid points in time.
        :param:lbc: left boundary condition.
        :type:lbc: class BC
        :param:rbc: right boundary condition.
        :type:rbc: class BC
        :param:h: time step.
        :param:algo: algorithm for calculation of RK4 coefficients.
        :param:scheme: RK4 scheme.
        '''
        self.R = R 
        self.Nr = Nr
        self.Nt = Nt
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
        Nt = self.Nt
        L = self._calc_operator() # the shape of coeff depends on BC
        N, N = L.shape
        dim = [Nt, N, N]
        self.E = np.zeros(dim)
        self.E2 = np.zeros(dim)
        self.Q = np.zeros(dim)
        self.f1 = np.zeros(dim)
        self.f2 = np.zeros(dim)
        self.f3 = np.zeros(dim)
        self.f4 = np.zeros(dim)
        self.f5 = np.zeros(dim)
        self.f6 = np.zeros(dim)
        for i in xrange(Nt):
            if i < Nt/2+1:
                kt = i 
            else:
                kt = i - Nt
            # R**(-2) for maping from [0,R] to [0,1]
            Lk = (L - np.diag((kt/self.r)**2)) / self.R**2
            self._calc_RK4_coeff(i, Lk)

    def _calc_operator(self):
        '''
            Currently, only symmetric boundary conditions are allowed, that is
                DBC-DBC
                RBC-RBC (including the special case NBC)
        '''
        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                D1t, D2t, r = cheb_D2_mat_dirichlet_dirichlet(self.Nr)
                r = r[1:-1]
            else:
                D1t, D2t, r = cheb_D2_mat_dirichlet_robin(self.Nr, 
                                                       self.rbc.beta)
                r = r[:-1]
        else:
            if self.rbc.kind == DIRICHLET:
                D1t, D2t, r = cheb_D2_mat_robin_dirichlet(self.Nr, 
                                                       self.lbc.beta)
                r = r[1:]
            else:
                D1t, D2t, r = cheb_D2_mat_robin_robin(self.Nr, 
                                                   self.lbc.beta,
                                                   self.rbc.beta)

        N, N = D2t.shape # N should be either Nr+1 or Nr-1
        self.r = r[:N/2].reshape(N/2) # reshape to vector
        D1 = D2t[:N/2,:N/2]
        D2 = D2t[:N/2,N-1:N/2-1:-1]
        E1 = D1t[:N/2,:N/2]
        E2 = D1t[:N/2,N-1:N/2-1:-1]
        MR = np.diag(1/self.r)
        #print self.r.shape, D1.shape, D2.shape, E1.shape, E2.shape, MR.shape
        L = (D1 + D2) + (np.dot(MR,E1) + np.dot(MR,E2))
        return L

    def _calc_RK4_coeff(self, i, L):
        L = self.c * L # the actual operator
        h = self.h
        c = 1.0
        M = 32; R = 15.;
        if self.scheme == 0:
            if self.algo == 0:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_nondiag(L, h, M, R)
            elif self.algo == 1:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_contour_hyperbolic(L, h, M)
            elif self.algo == 2:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_scale_square(L, h)
            else:
                raise ValueError('No such ETDRK4 coefficient algorithm!')
            self.E[i] = E[:,:]
            self.E2[i] = E2[:,:]
            self.Q[i] = Q[:,:]
            self.f1[i] = f1[:,:]
            self.f2[i] = f2[:,:]
            self.f3[i] = f3[:,:]
            f4 = None; f5 = None; f6 = None
        elif self.scheme == 1:
            if self.algo == 0:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_nondiag_krogstad(L, h, M, R)
            elif self.algo == 1:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_contour_hyperbolic_krogstad(L, h, c, M)
            elif self.algo == 2:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_scale_square_krogstad(L, h)
            else:
                raise ValueError('No such ETDRK4 coefficient algorithm!')
            self.E[i] = E[:,:]
            self.E2[i] = E2[:,:]
            Q = None
            self.f1[i] = f1[:,:]
            self.f2[i] = f2[:,:]
            self.f3[i] = f3[:,:]
            self.f4[i] = f4[:,:]
            self.f5[i] = f5[:,:]
            self.f6[i] = f6[:,:]
        else:
            raise ValueError('No such ETDRK4 scheme!')


    def solve(self, w, u0, q=None):
        '''
            w = w(theta, r)
            u0 = q(theta, r, t=0)
            q = q(theta, r, t)
            for r in (0, R] and t in [0,1].
            Discretization form:
                w(i,j), u0(i,j), q(i,j,t)
                i in [0, Nt-1]
                j in [0, (Nr+1)/2]
                t in [0, Ns]
        The particular order of theta, r is chosen to be compatible with
            etdrk4fxcy_scheme_krogstad
        which perform FFT in first dimension.
        '''
        u = u0.copy()
        E = self.E; E2 = self.E2; Q = self.Q
        f1 = self.f1; f2 = self.f2; f3 = self.f3
        f4 = self.f4; f5 = self.f5; f6 = self.f6
        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                v = u[:,1:-1]
                W = -w[:,1:-1]
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                Q, f1, f2, f3, q[:,:,1:-1])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                                   f4, f5, f6, q[:,:,1:-1])
                    else:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:,1:-1] = v
            else: # not allowed in current implementation.
                v = u[:,:-1]
                W = -w[:,:-1]
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                Q, f1, f2, f3, q[:,:,:-1])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                                   f4, f5, f6, q[:,:,:-1])
                    else:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:,:-1] = v
        else: # not allowed in current implementation.
            if self.rbc.kind == DIRICHLET:
                v = u[:,1:]
                W = -w[:,1:]
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                      Q, f1, f2, f3, q[:,:,1:])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                                   f4, f5, f6, q[:,:,1:])
                    else:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:,1:] = v
            else:
                v = u
                W = -w
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3, q)
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6, q)
                    else:
                        v = etdrk4fxcy_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u = v

        return u


class ETDRK4Cylind(object):
    def __init__(self, R, Lz, Nr, Nt, Nz, Ns, h=None, c=1.0,  
                 lbc=BC(), rbc=BC(), algo=1, scheme=1):
        '''
        The PDE is in the cylindrical coordinate,
            du/dt = cLu - wu
        where u=u(r,theta, z), L=d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/dtheta^2 +
        d^2/dz^2,
        w=w(r,theta, z), c is a constant. Domain is
            theta [0, 2*pi]
            r     [0, R]
            z     [0, Lz]
        First, do FFT in theta and z axes to obtain
            du(r,kt, kz)/dt = c L u(r,kt, kz) - Ftz[w(r,theta, z)u(r,theta, z)]
        where L = d^2/dr^2 + (1/r)d/dr - (1/r^2)kt^2 - kz^2
        See details in the Notebook (page 2013.8.16).

        The defaut left BC and right BC are RBCs.

        Test: None.

        :param:R: physical size of the radius of the cylinder.
        :param:Lz: physical size of the length of the cylinder.
        :param:Nr: r axis discretization, 0, 1, 2, ..., Nr. Nr must be ODD.
        :param:Nt: theta axis discretization, 0, 1, 2, ..., Nt-1
        :param:Nz: z axis discretization, 0, 1, 2, ..., Nz-1
        :param:Ns: number of grid points in time.
        :param:lbc: left boundary condition.
        :type:lbc: class BC
        :param:rbc: right boundary condition.
        :type:rbc: class BC
        :param:h: time step.
        :param:algo: algorithm for calculation of RK4 coefficients.
        :param:scheme: RK4 scheme.
        '''
        self.R = R 
        self.Lz = Lz
        self.Nr = Nr
        self.Nt = Nt
        self.Nz = Nz
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
        Nt = self.Nt
        Nz = self.Nz
        L = self._calc_operator() # the shape of coeff depends on BC
        N, N = L.shape
        I = np.eye(N)
        dim = [Nt, Nz, N, N]
        self.E = np.zeros(dim)
        self.E2 = np.zeros(dim)
        self.Q = np.zeros(dim)
        self.f1 = np.zeros(dim)
        self.f2 = np.zeros(dim)
        self.f3 = np.zeros(dim)
        self.f4 = np.zeros(dim)
        self.f5 = np.zeros(dim)
        self.f6 = np.zeros(dim)
        for i in xrange(Nt):
            for j in xrange(Nz):
                if i < Nt/2+1:
                    kt = i 
                else:
                    kt = i - Nt
                if j < Nz/2+1:
                    kz = j 
                else:
                    kz = j - Nz
                # R**(-2) for maping from [0,R] to [0,1]
                Lk = (L-np.diag((kt/self.r)**2))/self.R**2 
                Lk -= I*kz**2/self.Lz**2
                self._calc_RK4_coeff(i, j, Lk)

    def _calc_operator(self):
        '''
            Currently, only symmetric boundary conditions are allowed, that is
                DBC-DBC
                RBC-RBC (including the special case NBC)
        '''
        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                D1t, D2t, r = cheb_D2_mat_dirichlet_dirichlet(self.Nr)
                r = r[1:-1]
            else:
                D1t, D2t, r = cheb_D2_mat_dirichlet_robin(self.Nr, 
                                                       self.rbc.beta)
                r = r[:-1]
        else:
            if self.rbc.kind == DIRICHLET:
                D1t, D2t, r = cheb_D2_mat_robin_dirichlet(self.Nr, 
                                                       self.lbc.beta)
                r = r[1:]
            else:
                D1t, D2t, r = cheb_D2_mat_robin_robin(self.Nr, 
                                                   self.lbc.beta,
                                                   self.rbc.beta)

        N, N = D2t.shape # N should be either Nr+1 or Nr-1
        self.r = r[:N/2].reshape(N/2) # reshape to vector
        D1 = D2t[:N/2,:N/2]
        D2 = D2t[:N/2,N-1:N/2-1:-1]
        E1 = D1t[:N/2,:N/2]
        E2 = D1t[:N/2,N-1:N/2-1:-1]
        MR = np.diag(1/self.r)
        #print self.r.shape, D1.shape, D2.shape, E1.shape, E2.shape, MR.shape
        L = (D1 + D2) + (np.dot(MR,E1) + np.dot(MR,E2))
        return L

    def _calc_RK4_coeff(self, i, j, L):
        L = self.c * L # the actual operator
        h = self.h
        c = 1.0
        M = 32; R = 15.;
        if self.scheme == 0:
            if self.algo == 0:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_nondiag(L, h, M, R)
            elif self.algo == 1:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_contour_hyperbolic(L, h, M)
            elif self.algo == 2:
                E, E2, Q, f1, f2, f3 = \
                    etdrk4_coeff_scale_square(L, h)
            else:
                raise ValueError('No such ETDRK4 coefficient algorithm!')
            self.E[i,j] = E[:,:]
            self.E2[i,j] = E2[:,:]
            self.Q[i,j] = Q[:,:]
            self.f1[i,j] = f1[:,:]
            self.f2[i,j] = f2[:,:]
            self.f3[i,j] = f3[:,:]
            f4 = None; f5 = None; f6 = None
        elif self.scheme == 1:
            if self.algo == 0:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_nondiag_krogstad(L, h, M, R)
            elif self.algo == 1:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_contour_hyperbolic_krogstad(L, h, c, M)
            elif self.algo == 2:
                E, E2, f1, f2, f3, f4, f5, f6 = \
                    etdrk4_coeff_scale_square_krogstad(L, h)
            else:
                raise ValueError('No such ETDRK4 coefficient algorithm!')
            self.E[i,j] = E[:,:]
            self.E2[i,j] = E2[:,:]
            Q = None
            self.f1[i,j] = f1[:,:]
            self.f2[i,j] = f2[:,:]
            self.f3[i,j] = f3[:,:]
            self.f4[i,j] = f4[:,:]
            self.f5[i,j] = f5[:,:]
            self.f6[i,j] = f6[:,:]
        else:
            raise ValueError('No such ETDRK4 scheme!')


    def solve(self, w, u0, q=None):
        '''
            w = w(theta, z, r)
            u0 = q(theta, z, r, t=0)
            q = q(theta, z, r, t)
            for r in (0, R], z in [0, Lz], theta in [0, 2pi], and t in [0,1].
            Discretization form:
                w(i,j, k), u0(i,j, k), q(i,j,k,t)
                i in [0, Nt-1]
                j in [0, Nz-1]
                k in [0, (Nr+1)/2]
                t in [0, Ns]
        The particular order of theta, z, r is chosen to be compatible with
            etdrk4fxycz_scheme_krogstad
        which perform FFT in first two dimensions.
        '''
        u = u0.copy()
        E = self.E; E2 = self.E2; Q = self.Q
        f1 = self.f1; f2 = self.f2; f3 = self.f3
        f4 = self.f4; f5 = self.f5; f6 = self.f6
        if self.lbc.kind == DIRICHLET:
            if self.rbc.kind == DIRICHLET:
                v = u[:,:,1:-1]
                W = -w[:,:,1:-1]
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                Q, f1, f2, f3, q[:,:,1:-1])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                                   f4, f5, f6, q[:,:,1:-1])
                    else:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:,:,1:-1] = v
            else: # not allowed in current implementation.
                v = u[:,:,:-1]
                W = -w[:,:,:-1]
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                Q, f1, f2, f3, q[:,:,:-1])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                                   f4, f5, f6, q[:,:,:-1])
                    else:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:,:,:-1] = v
        else: # not allowed in current implementation.
            if self.rbc.kind == DIRICHLET:
                v = u[:,:,1:]
                W = -w[:,:,1:]
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, E, E2, 
                                                      Q, f1, f2, f3, q[:,:,1:])
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                                   f4, f5, f6, q[:,:,1:])
                    else:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u[:,:,1:] = v
            else:
                v = u
                W = -w
                if self.scheme == 0:
                    if q is not None:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3, q)
                    else:
                        v = etdrk4_scheme_coxmatthews(self.Ns, W, v, 
                                                  E, E2, Q, f1, f2, f3)
                else:
                    if q is not None:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6, q)
                    else:
                        v = etdrk4fxycz_scheme_krogstad(self.Ns, W, v, 
                                               E, E2, f1, f2, f3, 
                                               f4, f5, f6)
                u = v

        return u


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


def phi_contour_hyperbolic_old(z, l=0, M=32):
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


def phi_contour_hyperbolic(A, t=1, l=0, M=16):
    '''
    Evaluate \phi_l(tA) using complex contour integral methods with hyperbolic contour.
    See my Notebook page 2013.07.05.

    phi_l(z) = [phi_{l-1}(z) - phi_{l-1}(0)] / z, with
    phi_0(z) = exp(z)
    For example:
        phi_1(z) = [exp(z) - 1] / z
        phi_2(z) = [exp(z) - z - 1] / z^2
        phi_3(z) = [exp(z) - z^2/2 - z - 1] / z^3

    REF: 
        1. Schmelzer, T.; Trefethen, L. N. Electronic Transaction on Numerical
    Analysis, 2007, 29, 1-18.
        2. Weideman, J. A. C.; Trefethen, L. N. Mathematics of Computation,
        2007, 76, 1341-1356.
    '''

    N, N = A.shape
    I = np.eye(N)
    phi = 1j * np.zeros((N,N))

    alpha = 1.1721
    h = 1.0818 / M
    mu = 4.4921 * M / t
    k = np.arange(-M, M+1)
    u = k * h

    z = mu * (1 + np.sin(u*1j - alpha))
    v = np.cos(u*1j - alpha) # dz/du = i*mu*v

    if l == 0:
        c = np.exp(t*z) * v
    else:
        c = np.exp(t*z) * v / (t*z)**l 

    for i in np.arange(np.size(k)):
        sIA = inv(z[i] * I - A)
        phi += c[i] * sIA

    return np.real(0.5 * h * mu / np.pi * phi)


def etdrk4_coeff_contour_hyperbolic(L, h, M=16):
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
    Q = h * 0.5 * phi_contour_hyperbolic(L, 0.5*h, 1, M)
    phi1 = phi_contour_hyperbolic(L, h, 1, M)
    phi2 = phi_contour_hyperbolic(L, h, 2, M)
    phi3 = phi_contour_hyperbolic(L, h, 3, M)
    f1 = h * (phi1 - 3 * phi2 + 4 * phi3)
    f2 = h * 2 * (phi2 - 2 * phi3)
    f3 = h * (4 * phi3 - phi2)
    
    return E1, E2, Q, f1, f2, f3


def etdrk4_coeff_contour_hyperbolic_krogstad(L, h, c=1, M=16):
    '''
    Evaluate etdrk4 coefficients by complex contour integral using
    hyperbolic contour for Krogstad scheme.

    REF:
        * Schmelzer, T.; Trefethen, L. N. **Evaluating Matrix Functions for
        Exponential Integrators Via Caratheodory-Fejer Approximation and Contour Integrals* 2007.
        * Weideman, J. A.; Trefethen, L. N. "Parabolic and Hyperbolic Contours for Computing the Bromwich Integral" Math. Comput. 2007, 76, 1341.
        * Trefethen, L. N.; Weideman, J. A. C.; Schmelzer, T.; "Talbot Quadratures and Rational Approximations" BIT Numer. Math. 2006, 46, 653. 
    '''

    #E1 = phi_contour_hyperbolic(L, h*c, 0, M) # phi_0(h*L) = exp(h*L)
    E1 = expm(h*L)
    #E2 = phi_contour_hyperbolic(L, 0.5*h*c, 0, M) # phi_0(h/2*L)
    E2 = expm(0.5*h*L)
    f1 = h * 0.5 * phi_contour_hyperbolic(L, 0.5*h*c, 1, M)
    f2 = h * phi_contour_hyperbolic(L, 0.5*h*c, 2, M)
    phi1 = phi_contour_hyperbolic(L, h*c, 1, M)
    phi2 = phi_contour_hyperbolic(L, h*c, 2, M)
    phi3 = phi_contour_hyperbolic(L, h*c, 3, M)
    f3 = h * phi1
    f4 = h * 2 * phi2
    f5 = h * (4 * phi3 - phi2)
    f6 = -h * 4 * phi3
    
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


def etdrk4_scheme_coxmatthews(Ns, w, v, E, E2, Q, f1, f2, f3, q=None):
    '''
    Cox-Mattews ETDRK4, whose stiff order is 2.
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
        if q is not None:
            q[j+1] = v[:,0]

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


def etdrk4fxcy_scheme_krogstad(Ns, w, v, E, E2, f1, f2, f3, f4, f5, f6, q=None):
    '''
    w = w(x,y)
    v = v(x,y)
    The size of E, E2, f1, f2, f3, f4, f5, f6 is (Nx, Ny, Ny)
    FFT in x and Chebyshev in y.
    Krogstad ETDRK4, whose stiff order is 3 better than Cox-Matthews ETDRK4.

    Test: PASSED 2013.08.01
    '''
    vk = fft(v, axis=0)
    ak = np.zeros_like(vk)
    bk = np.zeros_like(vk)
    ck = np.zeros_like(vk)
    Nx, Ny = v.shape
    for s in xrange(Ns-1):
        vk = fft(v, axis=0)
        Nu = w * v
        Nuk = fft(Nu, axis=0)
        for i in xrange(Nx):
            ak[i] = np.dot(E2[i], vk[i]) + np.dot(f1[i], Nuk[i])
        a = ifft(ak, axis=0).real
        Na = w * a
        Nak = fft(Na, axis=0)
        for i in xrange(Nx):
            bk[i] = ak[i] + np.dot(f2[i], Nak[i]-Nuk[i])
        b = ifft(bk, axis=0).real
        Nb = w * b
        Nbk = fft(Nb, axis=0)
        for i in xrange(Nx):
            ck[i] = np.dot(E[i], vk[i]) + \
                    np.dot(f3[i], Nuk[i]) + \
                    np.dot(f4[i], Nbk[i]-Nuk[i])
        c = ifft(ck, axis=0).real
        Nc = w * c
        Nck = fft(Nc, axis=0)
        for i in xrange(Nx):
            vk[i] = ck[i] + np.dot(f4[i], Nak[i]) + \
                    np.dot(f5[i], Nuk[i]+Nck[i]) + \
                    np.dot(f6[i], Nak[i]+Nbk[i])
        v = ifft(vk, axis=0).real
        if q is not None:
            q[s+1] = v[:,:]

    return v


def etdrk4fxycz_scheme_krogstad(Ns, w, v, E, E2, f1, f2, f3, f4, f5, f6, q=None):
    '''
    w = w(x,y,z)
    v = v(x,y,z)
    The size of E, E2, f1, f2, f3, f4, f5, f6 is (Nx, Ny, Nz, Nz)
    FFT in x and y, Chebyshev in z.
    Krogstad ETDRK4, whose stiff order is 3 better than Cox-Matthews ETDRK4.
    '''
    #vk = fft2(v, axes=(0,1))
    ak = np.zeros_like(v)
    bk = np.zeros_like(v)
    ck = np.zeros_like(v)
    Nx, Ny, Nz = v.shape
    for s in xrange(Ns-1):
        vk = fft2(v, axes=(0,1))
        Nu = w * v
        Nuk = fft2(Nu, axes=(0,1))
        for i in xrange(Nx):
            for j in xrange(Ny):
                ak[i,j] = np.dot(E2[i,j], vk[i,j]) + np.dot(f1[i,j], Nuk[i,j])
        a = ifft2(ak, axes=(0,1)).real
        Na = w * a
        Nak = fft2(Na, axes=(0,1))
        for i in xrange(Nx):
            for j in xrange(Ny):
                bk[i,j] = ak[i,j] + np.dot(f2[i,j], Nak[i,j]-Nuk[i,j])
        b = ifft2(bk, axes=(0,1)).real
        Nb = w * b
        Nbk = fft2(Nb, axes=(0,1))
        for i in xrange(Nx):
            for j in xrange(Ny):
                ck[i] = np.dot(E[i,j], vk[i,j]) + \
                        np.dot(f3[i,j], Nuk[i,j]) + \
                        np.dot(f4[i,j], Nbk[i,j]-Nuk[i,j])
        c = ifft2(ck, axes=(0,1)).real
        Nc = w * c
        Nck = fft2(Nc, axes=(0,1))
        for i in xrange(Nx):
            for j in xrange(Ny):
                vk[i] = ck[i,j] + np.dot(f4[i,j], Nak[i,j]) + \
                        np.dot(f5[i,j], Nuk[i,j]+Nck[i,j]) + \
                        np.dot(f6[i,j], Nak[i,j]+Nbk[i,j])
        v = ifft2(vk, axes=(0,1)).real
        if q is not None:
            q[s+1] = v[:,:,:]

    return v



