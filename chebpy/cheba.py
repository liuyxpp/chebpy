# -*- coding: utf-8 -*-
"""
chebpy.cheba
============

Applications of Chebyshev spectral methods to solve PDEs.

"""

from time import time
from math import cos, acos
import numpy as np
from scipy.linalg import expm, expm2, expm3, inv
import matplotlib.pyplot as plt

from chebpy import cheb_fast_transform, cheb_inverse_fast_transform
from chebpy import cheb_D1_mat
from chebpy import etdrk4_coeff_ndiag

__all__ = ['cheb_mde_split',
           'cheb_mde_neumann_split',
           'cheb_mde_etdrk4',
           'cheb_mde_neumann_etdrk4',
           'cheb_mde_robin_etdrk4',
           'cheb_allen_cahn_etdrk4',
          ]

def cheb_mde_split(W, Lx, Ns):
    '''
    Solution of modified diffusion equation (MDE) 
    via Strang operator splitting as time-stepping scheme 
    and pseudospectral methods on Chebyshev grids.

    The MDE is:
        dq/dt = Dq + Wq
    where D is Laplace operator. Dirichlet boundary condition is assumed.
    '''

    ds = 1. / (Ns -1)
    N = np.size(W) - 1
    u = np.ones(N+1)
    u[0] = 0.; u[N] = 0.

    k2 = (np.pi * np.pi) / (Lx * Lx) * np.arange(N+1) * np.arange(N+1)
    expw = np.exp(-0.5 * ds * W)
    for i in xrange(Ns-1):
        u = expw * u
        ak = cheb_fast_transform(u) * np.exp(-ds * k2)
        u = cheb_inverse_fast_transform(ak)
        u = expw * u
        u[0] = 0.; u[N] = 0.

    ii = np.arange(N+1)
    x = 1. * ii * Lx / N
    return (u, x) 


def cheb_mde_neumann_split(W, Lx, Ns, boundary=0):
    '''
    Solution of modified diffusion equation (MDE) 
    via Strang operator splitting as time-stepping scheme 
    and pseudospectral methods on Chebyshev grids.

    The MDE is:
        dq/dt = Dq + Wq
    where D is Laplace operator. Neumann boundary condition is assumed.
    '''

    ds = 1. / (Ns -1)
    N = np.size(W) - 1
    u = np.ones(N+1)
    u[0] = 0.; u[N] = 0.

    k2 = (np.pi * np.pi) / (Lx * Lx) * np.arange(N+1) * np.arange(N+1)
    expw = np.exp(-0.5 * ds * W)
    for i in xrange(Ns-1):
        u = expw * u
        ak = cheb_fast_transform(u) * np.exp(-ds * k2)
        u = cheb_inverse_fast_transform(ak)
        u = expw * u

    ii = np.arange(N+1)
    x = 1. * ii * Lx / N
    return (u, x) 


def cheb_mde_etdrk4(W, Lx, Ns):
    '''
    Solution of modified diffusion equation (MDE) by ETDRK4 shceme.
    This method allows very large time step.
    
    Thus ETDRK4 is much faster than Strang Splitting or DCT.
    For example, when space is discretized in N = 256, then Splitting
    method needs at leat Ns = 1601 to achive the same accuracy of ETDRK4 
    with Ns = 11. This is remarkable. 

    The MDE is:
        dq/dt = Dq + Wq
        q(-1,t) = 0, t>=0
        q(+1,t) = 0, t>=0
        q(x,0) = 1, -1<x<1
    where D is Laplace operator.

    Computation is based on Chebyshev points, so linear term is
    non-diagonal.
    '''

    ds = 1. / (Ns-1)
    N = np.size(W) - 1
    D, xx = cheb_D1_mat(N)
    u = np.ones((N+1,1))
    u[0] = 0.; u[N] = 0.
    v = u[1:N]
    w = -W[1:N]
    w.shape = (N-1, 1)

    h = ds
    M = 32
    R = 15.
    L = np.dot(D, D) # L = D^2
    L = (4. / Lx**2) * L[1:N,1:N]
    Q, f1, f2, f3 = etdrk4_coeff_ndiag(L, h, M, R)

    A = h * L
    E = expm(A)
    E2 = expm(A/2)

    for j in xrange(Ns-1):
        Nu = w * v
        a = np.dot(E2, v) + np.dot(Q, Nu)
        Na = w * a
        b = np.dot(E2, v) + np.dot(Q, Na)
        Nb = w * b
        c = np.dot(E2, a) + np.dot(Q, 2*Nb-Nu)
        Nc = w * c
        v = np.dot(E, v) + np.dot(f1, Nu) + 2 * np.dot(f2, Na+Nb) + \
            np.dot(f3, Nc)

    u[1:N] = v[:]
    return (u, .5*(xx+1.)*Lx)


def cheb_mde_neumann_etdrk4(W, Lx, Ns):
    '''
    Solution of modified diffusion equation (MDE) with 
    Neumann boundary condition (NBC) by ETDRK4 shceme.
    NBC is also called Fixed flux boundary condition.

    The MDE is:
        dq/dt = Dq + Wq
        dq/dx |(x=-1) = 0, t>=0
        dq/dx |(x=+1) = 0, t>=0
        q(x,0) = 1, -1<x<1
    where D is Laplace operator.
    '''

    ds = 1. / (Ns-1)
    N = np.size(W) - 1
    D, xx = cheb_D1_mat(N)
    u = np.ones((N+1,1))
    v = u.copy()
    w = -W
    w.shape = (N+1, 1)

    h = ds
    M = 32
    R = 15.
    D1 = np.zeros_like(D)
    D1[1:N,:] = D[1:N,:]
    L = np.dot(D, D1) 
    L = (4. / Lx**2) * L
    Q, f1, f2, f3 = etdrk4_coeff_ndiag(L, h, M, R)

    A = h * L
    E = expm(A)
    E2 = expm(A/2)

    for j in xrange(Ns-1):
        Nu = w * v
        a = np.dot(E2, v) + np.dot(Q, Nu)
        Na = w * a
        b = np.dot(E2, v) + np.dot(Q, Na)
        Nb = w * b
        c = np.dot(E2, a) + np.dot(Q, 2*Nb-Nu)
        Nc = w * c
        v = np.dot(E, v) + np.dot(f1, Nu) + 2 * np.dot(f2, Na+Nb) + \
            np.dot(f3, Nc)

    return (v, .5*(xx+1.)*Lx)


def cheb_mde_robin_etdrk4_bak(W, Lx, Ns, ka, kb):
    '''
    Solution of modified diffusion equation (MDE) with 
    Neumann boundary condition (NBC) by ETDRK4 shceme.
    NBC is also called Fixed flux boundary condition.

    The MDE is:
        dq/dt = Dq + Wq
        kq + dq/dx = 0, x=+1 or x=-1, t>=0
        q(x,0) = 1, -1<x<1
    where D is Laplace operator.
    '''

    ds = 1. / (Ns-1)
    N = np.size(W) - 1
    D, xx = cheb_D1_mat(N)
    u = np.ones((N+1,1))
    v = u.copy()
    v[0] = 0.
    v[N] = 0.
    w = -W
    w.shape = (N+1, 1)

    h = ds
    M = 32
    R = 15.
    D1 = np.zeros_like(D)
    D1[1:N,:] = D[1:N,:]
    L = np.dot(D, D1) 
    L = (4. / Lx**2) * L
    Q, f1, f2, f3 = etdrk4_coeff_ndiag(L, h, M, R)

    A = h * L
    E = expm(A)
    E2 = expm(A/2)

    # Robin boundary
    d00 = D[0,0]; d0N = D[0,N]; dN0 = D[N,0]; dNN = D[N,N]
    li0 = L[1:N,0].copy(); liN = L[1:N,N].copy()
    li0.shape = (N-1, 1)
    liN.shape = (N-1, 1)
    kk = (ka + d00) * (kb + dNN) - d0N * dN0

    for j in xrange(Ns-1):
        Nu = w * v
        a = np.dot(E2, v) + np.dot(Q, Nu)

        sum0 = np.dot(D[0,1:N], a[1:N])
        sumN = np.dot(D[N,1:N], a[1:N])
        a[0] = (d0N * sumN - (kb + dNN) * sum0) / kk
        a[N] = (dN0 * sum0 - (ka + d00) * sumN) / kk
        Na = w * a
        b = np.dot(E2, v) + np.dot(Q, Na)

        sum0 = np.dot(D[0,1:N], b[1:N])
        sumN = np.dot(D[N,1:N], b[1:N])
        b[0] = (d0N * sumN - (kb + dNN) * sum0) / kk
        b[N] = (dN0 * sum0 - (ka + d00) * sumN) / kk
        Nb = w * b
        c = np.dot(E2, a) + np.dot(Q, 2*Nb-Nu)

        sum0 = np.dot(D[0,1:N], c[1:N])
        sumN = np.dot(D[N,1:N], c[1:N])
        c[0] = (d0N * sumN - (kb + dNN) * sum0) / kk
        c[N] = (dN0 * sum0 - (ka + d00) * sumN) / kk
        Nc = w * c
        v = np.dot(E, v) + np.dot(f1, Nu) + 2 * np.dot(f2, Na+Nb) + \
            np.dot(f3, Nc)
        sum0 = np.dot(D[0,1:N], v[1:N])
        sumN = np.dot(D[N,1:N], v[1:N])
        v[0] = (d0N * sumN - (kb + dNN) * sum0) / kk
        v[N] = (dN0 * sum0 - (ka + d00) * sumN) / kk

    #sum0 = np.dot(D[0,1:N], v[1:N])
    #sumN = np.dot(D[N,1:N], v[1:N])
    #v[0] = (d0N * sumN - (kb + dNN) * sum0) / kk
    #v[N] = (dN0 * sum0 - (ka + d00) * sumN) / kk
    return (v, .5*(xx+1.)*Lx)


def cheb_mde_robin_etdrk4(W, Lx, Ns, ka, kb):
    '''
    Solution of modified diffusion equation (MDE) with 
    Neumann boundary condition (NBC) by ETDRK4 shceme.
    NBC is also called Fixed flux boundary condition.

    The MDE is:
        dq/dt = Dq + Wq
        kq + dq/dx = 0, x=+1 or x=-1, t>=0
        q(x,0) = 1, -1<x<1
    where D is Laplace operator.
    '''

    ds = 1. / (Ns-1)
    N = np.size(W) - 1
    D, xx = cheb_D1_mat(N)
    u = np.ones((N+1,1))
    v = u[1:N]
    w = -W[1:N]
    w.shape = (N-1, 1)

    # Robin boundary
    d00 = D[0,0]; d0N = D[0,N]; dN0 = D[N,0]; dNN = D[N,N]
    D2 = np.dot(D, D)
    kk = (ka + d00) * (kb + dNN) - d0N * dN0
    L = np.zeros_like(D2)
    for i in xrange(1,N):
        for j in xrange(1,N):
            Xij = (d0N * D2[i,0] - (ka+d00) * D2[i,N]) / kk * D[N,j]
            Yij = (dN0 * D2[i,N] - (kb+dNN) * D2[i,0]) / kk * D[0,j]
            L[i,j] = D2[i,j] + Xij + Yij

    h = ds
    M = 32
    R = 15.
    L = (4. / Lx**2) * L[1:N,1:N]
    Q, f1, f2, f3 = etdrk4_coeff_ndiag(L, h, M, R)

    A = h * L
    E = expm(A)
    E2 = expm(A/2)

    for j in xrange(Ns-1):
        Nu = w * v

        a = np.dot(E2, v) + np.dot(Q, Nu)
        Na = w * a

        b = np.dot(E2, v) + np.dot(Q, Na)
        Nb = w * b

        c = np.dot(E2, a) + np.dot(Q, 2*Nb-Nu)
        Nc = w * c

        v = np.dot(E, v) + np.dot(f1, Nu) + 2 * np.dot(f2, Na+Nb) + \
            np.dot(f3, Nc)

    sum0 = np.dot(D[0,1:N], v)
    sumN = np.dot(D[N,1:N], v)
    u[0] = (d0N * sumN - (kb + dNN) * sum0) / kk
    u[N] = (dN0 * sum0 - (ka + d00) * sumN) / kk
    u[1:N] = v[:]
    return (u, .5*(xx+1.)*Lx)


def cheb_allen_cahn_etdrk4():
    '''
    Solution of Allen-Cahn equation by ETDRK4 shceme.

    u_t = eps*u_xx + u - u^3 on
        [-1,1], u(-1) = -1, u(1) = 1

    Computation is based on Chebyshev points, so linear term is
    non-diagonal.
    '''

    N = 80
    D, xx = cheb_D1_mat(N)
    x = xx[1:N]
    w = .53*x + .47*np.sin(-1.5*np.pi*x) - x
    u = np.concatenate(([[1]], w+x, [[-1]]))

    h = 1./4
    M = 32 # Number of points in upper half-plane
    kk = np.arange(1, M+1)
    # theta = pi/64 * {1, 3, 5, ..., 2*M-1}
    # the radius of the circular contour is 15.0
    r = 15.0 * np.exp(1j * np.pi * (kk - .5) / M)
    L = np.dot(D, D) # L = D^2
    L = .01 * L[1:N,1:N]
    A = h * L
    E = expm(A)
    E2 = expm(A/2)
    I = np.eye(N-1)
    Z = 1j * np.zeros((N-1,N-1))
    f1 = Z; f2 = Z; f3 = Z; Q = Z
    for j in xrange(M):
        z = r[j]
        zIA = inv(z * I - A)
        hzIA = h * zIA
        hzIAz2 = hzIA / z**2
        Q = Q + hzIA * (np.exp(z/2) - 1)
        f1 = f1 + hzIAz2 * (-4 - z + np.exp(z) * (4 - 3*z + z**2))
        f2 = f2 + hzIAz2 * (2 + z + np.exp(z) * (z - 2))
        f3 = f3 + hzIAz2 * (-4 - 3*z - z*z + np.exp(z) * (4 - z))
    f1 = np.real(f1 / M)
    f2 = np.real(f2 / M)
    f3 = np.real(f3 / M)
    Q = np.real(Q / M)

    tt = 0.
    tmax = 70
    nmax = int(round(tmax / h))
    nplt = int(round(5. / h))
    print nmax,nplt
    for n in xrange(nmax):
        t = (n+1) * h
        Nu = (w+x) - np.power(w+x, 3)
        a = np.dot(E2, w) + np.dot(Q, Nu)
        Na = (a+x) - np.power(a+x, 3)
        b = np.dot(E2, w) + np.dot(Q, Na)
        Nb = (b+x) - np.power(b+x, 3)
        c = np.dot(E2, a) + np.dot(Q, 2*Nb-Nu)
        Nc = (c+x) - np.power(c+x, 3)
        w = np.dot(E, w) + np.dot(f1, Nu) + 2 * np.dot(f2, Na+Nb) + \
            np.dot(f3, Nc)
        if ((n+1) % nplt) == 0:
            print n+1
            u = np.concatenate(([[1]],w+x,[[-1]])).T
            plt.plot(xx, u[0,:])
            plt.show()

