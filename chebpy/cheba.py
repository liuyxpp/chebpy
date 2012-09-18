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
from scipy.fftpack import dst, idst, dct, idct
import matplotlib.pyplot as plt

from chebpy import cheb_fast_transform, cheb_inverse_fast_transform
from chebpy import cheb_D1_mat, cheb_D2_mat_dirichlet_dirichlet
from chebpy import cheb_D2_mat_dirichlet_robin, cheb_D2_mat_robin_robin
from chebpy import etdrk4_coeff_nondiag, etdrk4_coeff_contour_hyperbolic
from chebpy import etdrk4_coeff_scale_square
from chebpy import etdrk4_scheme_coxmatthews, etdrk4_scheme_krogstad
from chebpy import etdrk4_coeff_nondiag_krogstad
from chebpy import etdrk4_coeff_contour_hyperbolic_krogstad
from chebpy import etdrk4_coeff_scale_square_krogstad
from chebpy import solve_tridiag_complex_thual

__all__ = ['cheb_mde_oss',
           'cheb_mde_osc',
           'cheb_mde_neumann_split',
           'cheb_mde_dirichlet_oscheb',
           'cheb_mde_neumann_oscheb',
           'cheb_mde_dirichlet_etdrk4',
           'cheb_mde_neumann_etdrk4',
           'cheb_mde_robin_etdrk4',
           'cheb_mde_robin_etdrk4_1',
           'cheb_mde_robin_etdrk4_2',
           'cheb_mde_mixed_etdrk4',
           'cheb_allen_cahn_etdrk4',
          ]

def cheb_mde_oss(W, u0, Lx, Ns):
    '''
    Solution of modified diffusion equation (MDE) 
    via Strang operator splitting as time-stepping scheme 
    and fast sine transform.

    The MDE is:
        dq/dt = Dq - Wq
    in the interval [0, L], with Dirichlet boundary conditions,
    where D is Laplace operator.
    Discretization:
        x_j = j*L_x/N, j = 0, 1, 2, ..., N

    :param:W: W_j, j = 0, 1, ..., N
    :param:u0: u0_j, j = 0, 1, ..., N
    :param:Lx: the physical length of the interval [0, L_x]
    :param:Ns: contour index, s_j, j = 0, 1, ..., N_s
    '''

    ds = 1. / (Ns - 1)
    N = np.size(u0) - 1
    v = np.zeros(N-1)
    v = u0[1:N] # v = {u[1], u[2], ..., u[N-1]}

    k2 = (np.pi/Lx)**2 * np.arange(1, N)**2
    expd = np.exp(-ds * k2)
    expw = np.exp(-0.5 * ds * W[1:N])
    for i in xrange(Ns-1):
        v = expw * v
        ak = dst(v, type=1) / N * expd
        v = 0.5 * idst(ak, type=1)
        v = expw * v

    ii = np.arange(N+1)
    x = 1. * ii * Lx / N
    u = u0.copy()
    u[1:N] = v
    u[0] = 0.; u[N] = 0.;
    return (u, x) 


def cheb_mde_osc(W, u0, Lx, Ns):
    '''
    Solution of modified diffusion equation (MDE) 
    via Strang operator splitting as time-stepping scheme 
    and fast cosine transform.

    The MDE is:
        dq/dt = Dq - Wq
    in the interval [0, L], with Neumann boundary conditions,
    where D is Laplace operator.
    Discretization:
        x_j = j*L_x/N, j = 0, 1, 2, ..., N

    :param:W: W_j, j = 0, 1, ..., N
    :param:u0: u0_j, j = 0, 1, ..., N
    :param:Lx: the physical length of the interval [0, L_x]
    :param:Ns: contour index, s_j, j = 0, 1, ..., N_s
    '''

    ds = 1. / (Ns - 1)
    N = np.size(u0) - 1
    u = u0.copy()

    k2 = (np.pi/Lx)**2 * np.arange(N+1)**2
    expd = np.exp(-ds * k2)
    expw = np.exp(-0.5 * ds * W)
    for i in xrange(Ns-1):
        u = expw * u
        ak = dct(u, type=1) / N * expd
        u = 0.5 * idct(ak, type=1)
        u = expw * u

    ii = np.arange(N+1)
    x = 1. * ii * Lx / N
    return (u, x) 


def cheb_mde_neumann_split(W, u0, Lx, Ns):
    '''
    Solution of modified diffusion equation (MDE) 
    via Strang operator splitting as time-stepping scheme 
    and pseudospectral methods on Chebyshev grids.

    The MDE is:
        dq/dt = Dq + Wq
    where D is Laplace operator. Neumann boundary condition is assumed.
    '''

    ds = 1. / (Ns - 1)
    N = np.size(W) - 1
    u = u0.copy()
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


def cheb_mde_oscheb(W, u0, Lx, Ns, bc_even, bc_odd):
    '''
    Solution of modified diffusion equation (MDE) 
    via Strang operator splitting as time-stepping scheme 
    and fast Chebyshev transform.

    Ref:
        Her, S. M.; Garcia-Cervera, C. J.; Fredrickson, G. H. macromolecules, 2012, 45, 2905.

    The MDE is:
        dq/dt = Dq - Wq
    in the interval [0, L], with homogeneous Dirichlet boundary conditions at both boundaries.
    where D is Laplace operator.
    Discretization:
        x_j = cos(j*L_x/N), j = 0, 1, 2, ..., N

    :param:W: W_j, j = 0, 1, ..., N
    :param:u0: u0_j, j = 0, 1, ..., N
    :param:Lx: the physical length of the interval [0, L_x]
    :param:Ns: contour index, s_j, j = 0, 1, ..., N_s
    '''
    ds = 1. / (Ns - 1)
    N = u0.size - 1
    u = u0.reshape(u0.size) # force 1D array
    K = (Lx/2.)**2
    lambp = K * (1. + 1j) / ds # \lambda_+
    lambn = K * (1. - 1j) / ds # \lambda_-

    c = np.ones(N+1)
    c[0] = 2.; c[-1] = 2. # c_0=c_N=2, c_1=c_2=...=c_{N-1}=1
    b = np.ones(N+3)
    b[N-1:] = 0. # b_{N-1}=...=b_{N+2}=0, b_0=b_1=...=b_{N-2}=1
    n = np.arange(N+1)
    pe = 0.25 * c[0:N-1:2] / n[2:N+1:2] / (n[2:N+1:2] - 1)
    po = 0.25 * c[1:N-1:2] / n[3:N+1:2] / (n[3:N+1:2] - 1)
    pep = -pe * lambp
    pen = -pe * lambn
    pop = -po * lambp
    pon = -po * lambn
    qe = 0.5 * b[2:N+1:2] / (n[2:N+1:2]**2 - 1) 
    qo = 0.5 * b[3:N+1:2] / (n[3:N+1:2]**2 - 1)
    qep = 1 + qe * lambp
    qen = 1 + qe * lambn
    qop = 1 + qo * lambp
    qon = 1 + qo * lambn
    re = 0.25 * b[4:N+3:2] / n[2:N+1:2] / (n[2:N+1:2] + 1)
    ro = 0.25 * b[5:N+3:2] / n[3:N+1:2] / (n[3:N+1:2] + 1)
    rep = -re * lambp
    ren = -re * lambn
    rop = -ro * lambp
    ron = -ro * lambn
    pg = np.zeros(pe.size + po.size)
    pg[0:N-1:2] = pe
    pg[1:N-1:2] = po
    qg = np.zeros(qe.size + qo.size)
    qg[0:N-1:2] = qe
    qg[1:N-1:2] = qo
    rg = np.zeros(re.size + ro.size)
    rg[0:N-1:2] = re
    rg[1:N-1:2] = ro

    #dbce = np.ones(pe.size + 1)
    #dbco = np.ones(po.size + 1)
    dbce = bc_even
    dbco = bc_odd
    f = np.zeros(u.size + 2) # u.size is N+1
    ge = np.zeros(pe.size + 1)
    go = np.zeros(po.size + 1)

    uc = u.astype(complex)
    fc = f.astype(complex)
    gec = ge.astype(complex)
    goc = go.astype(complex)
    
    expw = np.exp(-0.5 * ds * W)
    for i in xrange(Ns-1):
        u = expw * u

        u = cheb_fast_transform(u)

        f[:N+1] = u
        g = pg * f[0:N-1] - qg * f[2:N+1] + rg * f[4:N+3]
        ge[1:] = g[0:N-1:2]
        go[1:] = g[1:N-1:2]
        ue = solve_tridiag_complex_thual(pen, qen, ren, dbce, ge)
        uo = solve_tridiag_complex_thual(pon, qon, ron, dbco, go)
        uc[0:N+1:2] = ue
        uc[1:N+1:2] = uo

        fc[:N+1] = uc
        gc = pg * fc[0:N-1] - qg * fc[2:N+1] + rg * fc[4:N+3]
        gec[1:] = gc[0:N-1:2]
        goc[1:] = gc[1:N-1:2]
        ue = solve_tridiag_complex_thual(pep, qep, rep, dbce, gec)
        uo = solve_tridiag_complex_thual(pop, qop, rop, dbco, goc)
        uc[0:N+1:2] = ue
        uc[1:N+1:2] = uo
        u = uc.real

        u = cheb_inverse_fast_transform(u)

        u = 2. * (K/ds)**2 * expw * u

    return u


def cheb_mde_dirichlet_oscheb(W, u0, Lx, Ns):
    '''
    Solution of modified diffusion equation (MDE) 
    via Strang operator splitting as time-stepping scheme 
    and fast Chebyshev transform.

    Ref:
        Her, S. M.; Garcia-Cervera, C. J.; Fredrickson, G. H. macromolecules, 2012, 45, 2905.

    The MDE is:
        dq/dt = Dq - Wq
    in the interval [0, L], with homogeneous Dirichlet boundary conditions at both boundaries.
    where D is Laplace operator.
    Discretization:
        x_j = cos(j*L_x/N), j = 0, 1, 2, ..., N

    :param:W: W_j, j = 0, 1, ..., N
    :param:u0: u0_j, j = 0, 1, ..., N
    :param:Lx: the physical length of the interval [0, L_x]
    :param:Ns: contour index, s_j, j = 0, 1, ..., N_s
    '''
    N = u0.size - 1 # 0, 1, ..., N
    if N % 2 == 0:
        Ne = N/2 + 1 # even index: 0, 2, ..., N
    else:
        Ne = (N-1)/2 + 1 # even index: 0, 2, ..., N-1
    No = (N+1) - Ne
    dbce = np.ones(Ne)
    dbco = np.ones(No)

    return cheb_mde_oscheb(W, u0, Lx, Ns, dbce, dbco)


def cheb_mde_neumann_oscheb(W, u0, Lx, Ns):
    '''
    Solution of modified diffusion equation (MDE) 
    via Strang operator splitting as time-stepping scheme 
    and fast Chebyshev transform.

    Ref:
        Her, S. M.; Garcia-Cervera, C. J.; Fredrickson, G. H. macromolecules, 2012, 45, 2905.

    The MDE is:
        dq/dt = Dq - Wq
    in the interval [0, L], subjects to homogeneous Neumann boundary conditions at both boundaries,
    where D is Laplace operator.
    Discretization:
        x_j = cos(j*L_x/N), j = 0, 1, 2, ..., N

    :param:W: W_j, j = 0, 1, ..., N
    :param:u0: u0_j, j = 0, 1, ..., N
    :param:Lx: the physical length of the interval [0, L_x]
    :param:Ns: contour index, s_j, j = 0, 1, ..., N_s
    '''
    N = u0.size - 1
    if N % 2 == 0:
        Ne = N/2 + 1 # even index: 0, 2, ..., N
    else:
        Ne = (N-1)/2 + 1 # even index: 0, 2, ..., N-1
    No = (N+1) - Ne
    nbce = np.arange(Ne)**2
    nbco = np.arange(No)**2

    return cheb_mde_oscheb(W, u0, Lx, Ns, nbce, nbco)


def cheb_mde_dirichlet_etdrk4(W, u0, Lx, Ns, algo=0, scheme=0):
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
        q(x,0) = u0, -1<x<1
    where D is Laplace operator.

    Computation is based on Chebyshev points, so linear term is
    non-diagonal.
    '''

    ds = 1. / (Ns-1)
    N = np.size(W) - 1
    #D, xx = cheb_D1_mat(N)
    #u = np.ones((N+1,1))
    u = u0.copy()
    u.shape = (N+1, 1)
    u[0] = 0.; u[N] = 0. # Dirichlet boundary condition
    v = u[1:N]
    w = -W[1:N]
    w.shape = (N-1, 1)

    h = ds
    #L = np.dot(D, D) # L = D^2
    D1t, L, x = cheb_D2_mat_dirichlet_dirichlet(N)
    L = (4. / Lx**2) * L

    M = 32
    R = 15.
    
    if scheme == 0:
        if algo == 0:
            E, E2, Q, f1, f2, f3 = etdrk4_coeff_nondiag(L, h, M, R)
        elif algo == 1:
            E, E2, Q, f1, f2, f3 = etdrk4_coeff_contour_hyperbolic(L, h, M)
        elif algo == 2:
            E, E2, Q, f1, f2, f3 = etdrk4_coeff_scale_square(L, h)
        else:
            E, E2, Q, f1, f2, f3 = etdrk4_coeff_nondiag(L, h, M, R)
        v = etdrk4_scheme_coxmatthews(Ns, w, v, E, E2, Q, f1, f2, f3)
    elif scheme == 1:
        if algo == 0:
            E, E2, f1, f2, f3, f4, f5, f6 = etdrk4_coeff_nondiag_krogstad(L, h, M, R)
        elif algo == 1:
            E, E2, f1, f2, f3, f4, f5, f6 = etdrk4_coeff_contour_hyperbolic_krogstad(L, h, M)
        elif algo == 2:
            E, E2, f1, f2, f3, f4, f5, f6 = etdrk4_coeff_scale_square_krogstad(L, h)
        else:
            E, E2, f1, f2, f3, f4, f5, f6 = etdrk4_coeff_nondiag_krogstad(L, h, M, R)
        v = etdrk4_scheme_krogstad(Ns, w, v, E, E2, f1, f2, f3, f4, f5, f6)
    else:
        raise ValueError('No such ETDRK4 scheme!')

    u[1:N] = v[:]
    return (u, .5*(x+1.)*Lx)


def cheb_mde_neumann_etdrk4(W, u0, Lx, Ns):
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
    u = u0.copy()
    u.shape = (N+1,1)
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
    Q, f1, f2, f3 = etdrk4_coeff_nondiag(L, h, M, R)

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


def cheb_mde_robin_etdrk4_1(W, u0, Lx, Ns, ka, kb):
    '''
    Solution of modified diffusion equation (MDE) with 
    Neumann boundary condition (NBC) by ETDRK4 shceme.
    NBC is also called Fixed flux boundary condition.

    The MDE is:
        dq/dt = Dq + Wq
        kq + dq/dx = 0, x=+1(kb) or x=-1(ka), t>=0
        q(x,0) = 1, -1<x<1
    where D is Laplace operator.
    '''

    ds = 1. / (Ns-1)
    N = np.size(W) - 1
    D, xx = cheb_D1_mat(N)
    u = u0.copy()
    u.shape = (N+1,1)
    v = u[1:N]
    w = -W[1:N]
    w.shape = (N-1, 1)

    # Robin boundary
    d00 = D[0,0]; d0N = D[0,N]; dN0 = D[N,0]; dNN = D[N,N]
    D2 = np.dot(D, D)
    kk = (kb + d00) * (ka + dNN) - d0N * dN0
    L = np.zeros_like(D2)
    for i in xrange(1,N):
        for j in xrange(1,N):
            Xij = (d0N * D2[i,0] - (kb+d00) * D2[i,N]) / kk * D[N,j]
            Yij = (dN0 * D2[i,N] - (ka+dNN) * D2[i,0]) / kk * D[0,j]
            L[i,j] = D2[i,j] + Xij + Yij

    h = ds
    M = 32
    R = 15.
    L = (4. / Lx**2) * L[1:N,1:N]
    Q, f1, f2, f3 = etdrk4_coeff_nondiag(L, h, M, R)

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
    u[0] = (d0N * sumN - (ka + dNN) * sum0) / kk
    u[N] = (dN0 * sum0 - (kb + d00) * sumN) / kk
    u[1:N] = v[:]
    return (u, .5*(xx+1.)*Lx)


def cheb_mde_robin_etdrk4_3(W, u0, Lx, Ns, ka, kb):
    '''
    This produce incorrect result. Should not be used.
    '''

    ds = 1. / (Ns-1)
    N = np.size(W) - 1
    D, xx = cheb_D1_mat(N)
    u = u0.copy()
    u.shape = (N+1,1)
    v = u[1:N]
    w = -W[1:N]
    w.shape = (N-1, 1)

    # Robin boundary
    d00 = D[0,0]; d0N = D[0,N]; dN0 = D[N,0]; dNN = D[N,N]
    D2 = np.dot(D, D)
    kk = (ka + d00) * (kb + dNN) - d0N * dN0
    k1 = (dN0 - kb - dNN) / kk
    k2 = (d0N - ka - d00) / kk
    L = np.zeros_like(D2)
    for i in xrange(1,N):
        sumDij = np.sum(D[i,:])
        for k in xrange(1,N):
            L[i,k] = D2[i,k] + (k1 * D[i,0] * D[0,k] + 
                                k2 * D[i,N] * D[N,k]) * sumDij

    h = ds
    M = 32
    R = 15.
    L = (4. / Lx**2) * L[1:N,1:N]
    Q, f1, f2, f3 = etdrk4_coeff_nondiag(L, h, M, R)

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


def cheb_mde_robin_etdrk4_2(W, u0, Lx, Ns, ka, kb):
    ds = 1. / (Ns-1)
    N = np.size(W) - 1
    u = u0.copy()
    u.shape = (N+1,1)
    v = u.copy()
    w = -W
    w.shape = (N+1, 1)

    # Robin boundary
    D1t, L, xx = cheb_D2_mat_robin_robin(N, ka, kb)

    h = ds
    M = 32
    R = 15.
    L = (4. / Lx**2) * L
    Q, f1, f2, f3 = etdrk4_coeff_nondiag(L, h, M, R)

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


def cheb_mde_robin_etdrk4(W, u0, Lx, Ns, ka, kb):
    '''
    Solution of modified diffusion equation (MDE) with 
    Robin boundary condition (RBC) by ETDRK4 shceme.

    The MDE is:
        dq/dt = Dq + Wq
        kq + dq/dx = 0, x=+1 or x=-1, t>=0
        q(x,0) = 1, -1<x<1
    where D is Laplace operator.

    The disrete matrix for Laplace operator is
        sum_{k=0}^N L_{ik} u_k
    where
        L_{ik} = D_{ij} * D1_{jk}, for i = 0, 1, ..., N; k = 1, 2, ..., N-1
    with
        D1_{jk} = D_{jk}    for j = 1, 2, ..., N-1; k = 0, 1, ..., N
        D1_{jk} = 0         for j = 0 or j = N; k = 0, 1, ..., N
    and
        L-{i0} = D_{ij} * D1_{j0} - ka * D_{i0}
        L-{iN} = D_{ij} * D1_{jN} - ka * D_{iN}

    '''

    ds = 1. / (Ns-1)
    N = np.size(W) - 1
    D, xx = cheb_D1_mat(N)
    u = u0.copy()
    u.shape = (N+1,1)
    v = u.copy()
    w = -W
    w.shape = (N+1, 1)

    h = ds
    M = 32
    R = 15.
    D1 = np.zeros_like(D)
    D1[1:N,:] = D[1:N,:]
    L = np.dot(D, D1) 
    L[:,0] -= D[:,0] * kb
    L[:,N] -= D[:,N] * ka
    L = (4. / Lx**2) * L
    Q, f1, f2, f3 = etdrk4_coeff_nondiag(L, h, M, R)

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


def cheb_mde_mixed_etdrk4(W, u0, Lx, Ns, kb):
    '''
    Solution of modified diffusion equation (MDE) with 
    mixed DBC-RBC by ETDRK4 shceme.

    The MDE is:
        dq/dt = Dq + Wq
    where D is Laplace operator.
    For RBC-DBC
        kq + dq/dx = 0, x=-1 (ka), t>=0
        q = 0, x=+1, t>=0
        q(x,0) = 1, -1<x<1
    The disrete matrix for Laplace operator is
        sum_{k=1}^N L_{ik} u_k
    where
        L_{ik} = D_{ij} * D1_{jk}, for i = 0, 1, ..., N; k = 1, 2, ..., N-1
    with
        D1_{jk} = D_{jk}    for j = 1, 2, ..., N-1; k = 0, 1, ..., N
        D1_{jk} = 0         for j = N; k = 0, 1, ..., N
    and
        L_{iN} = D_{ij} * D1_{jN} - ka * D_{iN}

    For DBC-RBC
        q = 0, x=-1, t>=0
        kq + dq/dx = 0, x=+1, t>=0
        q(x,0) = 1, -1<x<1

    '''

    ds = 1. / (Ns-1)
    N = np.size(W) - 1
    u = u0.copy()
    u.shape = (N+1,1)
    v = np.zeros(N)
    v = u[:-1]
    v.shape = (N, 1)
    w = -W[:-1]
    w.shape = (N, 1)

    h = ds
    M = 32
    R = 15.
    D1, L, xx = cheb_D2_mat_dirichlet_robin(N, kb)
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

    u[:-1] = v
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

