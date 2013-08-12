# -*- coding: utf-8 -*-
"""
chebpy.osf
==========

OSS and OSC class.

"""

import numpy as np
from scipy.fftpack import dst, idst, dct, idct
from scipy.fftpack import fft, ifft, fft2, ifft2, fftn, ifftn

__all__ = ['OSS', # Operator Splitting, Sine basis
           'OSC', # Operator Splitting, Cosine basis
           'OSF', # Operator splitting, Fourier basis, 1D
           'OSF2d', # Operator splitting, Fourier basis, 2D
           'OSF3d', # Operator splitting, Fourier basis, 3D
          ]

class OSS(object):
    def __init__(self, Lx, N, Ns, h=None):
        '''
        :param:Lx: physical size of the 1D spacial grid.
        :param:Ns: number of grid points in time.
        :param:N: number of grid points in space.
        :param:h: time step.
        '''
        self.Lx = Lx
        self.N = N
        self.Ns = Ns
        if h is None:
            self.h = 1. / (Ns - 1)
        else:
            self.h = h
        
        self.update()

    def update(self):
        ii = np.arange(self.N+1)
        self.x = 1. * ii * self.Lx / self.N
        k2 = (np.pi/self.Lx)**2 * np.arange(1, self.N)**2
        self.expd = np.exp(-self.h * k2)

    def solve(self, w, u0, q=None):
        '''
            dq/dt = Dq + Wq = Dq - wq
        '''
        u = u0.copy()
        v = u[1:-1] # v = {u[1], u[2], ..., u[N-1]}
        expw = np.exp(-0.5 * self.h * w[1:-1])
        for i in xrange(self.Ns-1):
            v = expw * v
            ak = dst(v, type=1) / self.N * self.expd
            v = 0.5 * idst(ak, type=1)
            v = expw * v
            if q is not None:
                q[i+1, 1:-1] = v

        u[1:-1] = v
        u[0] = 0.; u[-1] = 0.;

        return (u, self.x)


class OSC(object):
    def __init__(self, Lx, N, Ns, h=None):
        '''
        :param:Lx: physical size of the 1D spacial grid.
        :param:Ns: number of grid points in time.
        :param:N: number of grid points in space.
        :param:h: time step.
        '''
        self.Lx = Lx
        self.N = N
        self.Ns = Ns
        if h is None:
            self.h = 1. / (Ns - 1)
        else:
            self.h = h
        
        self.update()

    def update(self):
        ii = np.arange(self.N+1)
        self.x = 1. * ii * self.Lx / self.N
        k2 = (np.pi/self.Lx)**2 * np.arange(self.N+1)**2
        self.expd = np.exp(-self.h * k2)

    def solve(self, w, u0, q=None):
        '''
            dq/dt = Dq + Wq = Dq - wq
        '''
        u = u0.copy()
        expw = np.exp(-0.5 * self.h * w)
        for i in xrange(self.Ns-1):
            u = expw * u
            ak = dct(u, type=1) / self.N * self.expd
            u = 0.5 * idct(ak, type=1)
            u = expw * u
            if q is not None:
                q[i+1] = u

        return (u, self.x)



class OSF(object):
    def __init__(self, Lx, N, Ns, h=None):
        '''
        :param:Lx: physical size of the 1D spacial grid.
        :param:N: number of grid points in space.
        :param:Ns: number of grid points in time.
        :param:h: time step.
        '''
        self.Lx = Lx
        self.N = N
        self.Ns = Ns
        if h is None:
            self.h = 1. / (Ns - 1)
        else:
            self.h = h
        
        self.update()

    def update(self):
        Lx = self.Lx
        N = self.N
        h = self.h
        k2 = [i**2 for i in xrange(N/2+1)] # i=0, 1, ..., N/2
        k2.extend([(N-i)**2 for i in xrange(N/2+1, N)]) # i=N/2+1, ..., N-1
        k2 = np.array(k2) * (2*np.pi/Lx)**2  
        self.expd = np.exp(-h * k2)

    def solve(self, w, u0, q=None):
        '''
            dq/dt = Dq + Wq = Dq - wq
        '''
        u = u0.copy()
        h = self.h
        expw = np.exp(-0.5 * h * w)
        for i in xrange(self.Ns-1):
            u = expw * u
            ak = fft(u) * self.expd
            u = ifft(ak).real
            u = expw * u
            if q is not None:
                q[i+1, :] = u

        return u




class OSF2d(object):
    def __init__(self, Lx, Ly, Nx, Ny, Ns, h=None):
        '''
        :param:Lx: physical size of the 1D spacial grid.
        :param:N: number of grid points in space.
        :param:Ns: number of grid points in time.
        :param:h: time step.
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
        
        self.update()

    def update(self):
        Lx = self.Lx
        Ly = self.Ly
        Nx = self.Nx
        Ny = self.Ny
        h = self.h
        ccx = (2*np.pi/Lx)**2
        ccy = (2*np.pi/Ly)**2
        k2 = np.zeros((Nx,Ny))
        for i in xrange(Nx):
            for j in xrange(Ny):
                if i < Nx/2+1:
                    kx2 = i**2
                else:
                    kx2 = (Nx-i)**2
                if j < Ny/2+1:
                    ky2 = j**2
                else:
                    ky2 = (Ny-j)**2
                k2[i,j] = ccx * kx2 + ccy * ky2
        self.expd = np.exp(-h * k2)

    def solve(self, w, u0, q=None):
        '''
            dq/dt = Dq + Wq = Dq - wq
        '''
        u = u0.copy()
        h = self.h
        expw = np.exp(-0.5 * h * w)
        for i in xrange(self.Ns-1):
            u = expw * u
            ak = fft2(u) * self.expd
            u = ifft2(ak).real
            u = expw * u
            if q is not None:
                q[i+1] = u

        return u




class OSF3d(object):
    def __init__(self, Lx, Ly, Lz, Nx, Ny, Nz, Ns, h=None):
        '''
        :param:Lx: physical size of the 1D spacial grid.
        :param:N: number of grid points in space.
        :param:Ns: number of grid points in time.
        :param:h: time step.
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
        
        self.update()

    def update(self):
        Lx = self.Lx
        Ly = self.Ly
        Lz = self.Lz
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        h = self.h
        ccx = (2*np.pi/Lx)**2
        ccy = (2*np.pi/Ly)**2
        ccz = (2*np.pi/Lz)**2
        k2 = np.zeros((Nx,Ny,Nz))
        for i in xrange(Nx):
            for j in xrange(Ny):
                for k in xrange(Nz):
                    if i < Nx/2+1:
                        kx2 = i**2
                    else:
                        kx2 = (Nx-i)**2
                    if j < Ny/2+1:
                        ky2 = j**2
                    else:
                        ky2 = (Ny-j)**2
                    if k < Nz/2+1:
                        kz2 = k**2
                    else:
                        kz2 = (Nz-k)**2
                    k2[i,j,k] = ccx * kx2 + ccy * ky2 + ccz * kz2
        self.expd = np.exp(-h * k2)

    def solve(self, w, u0, q=None):
        '''
            dq/dt = Dq + Wq = Dq - wq
        '''
        u = u0.copy()
        h = self.h
        expw = np.exp(-0.5 * h * w)
        for i in xrange(self.Ns-1):
            u = expw * u
            ak = fftn(u) * self.expd
            u = ifftn(ak).real
            u = expw * u
            if q is not None:
                q[i+1] = u

        return u



