# -*- coding: utf-8 -*-
"""
chebpy.oscheb
=============

OSCHEB class.

"""

import numpy as np
from scipy.fftpack import dst, idst, dct, idct

__all__ = ['OSS', # Operator Split Sine Class
           'OSC', # Operator Split Cosine Class
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


