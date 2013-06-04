# -*- coding: utf-8 -*-
"""
chebpy.oscheb
=============

OSCHEB class.

"""

import numpy as np

from chebpy import BC, DIRICHLET, NEUMANN
from chebpy import solve_tridiag_complex_thual
from chebpy import cheb_fast_transform, cheb_inverse_fast_transform

__all__ = ['OSCHEB', # Operator Split Chebyshev series
          ]

class OSCHEB(object):
    def __init__(self, Lx, N, Ns, h=None, bc=BC()):
        '''
        Both BCs are DBCs or NBCs. The default is DBC.

        :param:Lx: physical size of the 1D spacial grid.
        :param:Ns: number of grid points in time.
        :param:N: number of grid points in space.
        :param:h: time step.
        '''
        self.Lx = Lx
        self.N = N
        self.Ns = Ns
        self.bc = bc
        if h is None:
            self.h = 1. / (Ns - 1)
        else:
            self.h = h
        
        self.update()

    def update(self):
        ii = np.arange(self.N+1)
        self.x = np.cos(np.pi * ii / self.N)
        self.x = .5 * (self.x + 1) * self.Lx
        self._boundary_layer()
        self._calc_coefficients()

    def _boundary_layer(self):
        N = self.N
        if N % 2 == 0:
            Ne = N/2 + 1 # even index: 0, 2, ..., N
        else:
            Ne = (N-1)/2 + 1 # even index: 0, 2, ..., N-1
        No = (N+1) - Ne
        if self.bc.kind == DIRICHLET:
            self.bce = np.ones(Ne)
            self.bco = np.ones(No)
        elif self.bc.kind == NEUMANN:
            self.bce = np.arange(Ne)**2
            self.bco = np.arange(No)**2
        else:
            raise ValueError('BCs other than DBC and NBC'
                             'are not supported.')

    def _calc_coefficients(self):
        N = self.N
        self.K = (self.Lx / 2.0)**2
        lambp = self.K * (1. + 1j) / self.h # \lambda_+
        lambn = self.K * (1. - 1j) / self.h # \lambda_-

        c = np.ones(N+1)
        c[0] = 2.; c[-1] = 2. # c_0=c_N=2, c_1=c_2=...=c_{N-1}=1
        b = np.ones(N+3)
        b[N-1:] = 0. # b_{N-1}=...=b_{N+2}=0, b_0=b_1=...=b_{N-2}=1
        n = np.arange(N+1)
        pe = 0.25 * c[0:N-1:2] / n[2:N+1:2] / (n[2:N+1:2] - 1)
        po = 0.25 * c[1:N-1:2] / n[3:N+1:2] / (n[3:N+1:2] - 1)
        self.pep = -pe * lambp
        self.pen = -pe * lambn
        self.pop = -po * lambp
        self.pon = -po * lambn
        qe = 0.5 * b[2:N+1:2] / (n[2:N+1:2]**2 - 1) 
        qo = 0.5 * b[3:N+1:2] / (n[3:N+1:2]**2 - 1)
        self.qep = 1 + qe * lambp
        self.qen = 1 + qe * lambn
        self.qop = 1 + qo * lambp
        self.qon = 1 + qo * lambn
        re = 0.25 * b[4:N+3:2] / n[2:N+1:2] / (n[2:N+1:2] + 1)
        ro = 0.25 * b[5:N+3:2] / n[3:N+1:2] / (n[3:N+1:2] + 1)
        self.rep = -re * lambp
        self.ren = -re * lambn
        self.rop = -ro * lambp
        self.ron = -ro * lambn
        self.pg = np.zeros(pe.size + po.size)
        self.pg[0:N-1:2] = pe
        self.pg[1:N-1:2] = po
        self.qg = np.zeros(qe.size + qo.size)
        self.qg[0:N-1:2] = qe
        self.qg[1:N-1:2] = qo
        self.rg = np.zeros(re.size + ro.size)
        self.rg[0:N-1:2] = re
        self.rg[1:N-1:2] = ro

    def solve(self, w, u0, q=None):
        '''
            dq/dt = Dq + Wq = Dq - wq
        '''
        N = self.N
        u = u0.reshape(u0.size) # force 1D array
        f = np.zeros(u.size + 2) # u.size is N+1
        ge = np.zeros(self.pep.size + 1)
        go = np.zeros(self.pop.size + 1)
        uc = u.astype(complex)
        fc = f.astype(complex)
        gec = ge.astype(complex)
        goc = go.astype(complex)

        expw = np.exp(-0.5 * self.h * w)
        for i in xrange(self.Ns-1):
            u = expw * u

            u = cheb_fast_transform(u)

            f[:N+1] = u
            g = self.pg * f[0:N-1] - self.qg * f[2:N+1] \
                    + self.rg * f[4:N+3]
            ge[1:] = g[0:N-1:2]
            go[1:] = g[1:N-1:2]
            ue = solve_tridiag_complex_thual(self.pen, self.qen, self.ren,
                                             self.bce, ge)
            uo = solve_tridiag_complex_thual(self.pon, self.qon, self.ron,
                                             self.bco, go)
            uc[0:N+1:2] = ue
            uc[1:N+1:2] = uo

            fc[:N+1] = uc
            gc = self.pg * fc[0:N-1] - self.qg * fc[2:N+1] \
                    + self.rg * fc[4:N+3]
            gec[1:] = gc[0:N-1:2]
            goc[1:] = gc[1:N-1:2]
            ue = solve_tridiag_complex_thual(self.pep, self.qep, self.rep,
                                             self.bce, gec)
            uo = solve_tridiag_complex_thual(self.pop, self.qop, self.rop,
                                             self.bco, goc)
            uc[0:N+1:2] = ue
            uc[1:N+1:2] = uo
            u = uc.real

            u = cheb_inverse_fast_transform(u)

            u = 2. * (self.K/self.h)**2 * expw * u
            
            if q is not None:
                q[i+1] = u

        return (u, self.x)


