# -*- coding: utf-8 -*-
#/usr/bin/env python
"""
test_mde
========

Tests of solution of modiffied diffusion equation (MDE).

"""

from time import time
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

from chebpy import cheb_mde_oss, cheb_mde_osc
from chebpy import cheb_mde_neumann_split
from chebpy import cheb_mde_etdrk4
from chebpy import cheb_mde_neumann_etdrk4
from chebpy import cheb_mde_robin_etdrk4, cheb_mde_robin_etdrk4_1
from chebpy import cheb_mde_robin_etdrk4_2, cheb_mde_robin_etdrk4_3
from chebpy import cheb_mde_mixed_etdrk4
from chebpy import cheb_allen_cahn_etdrk4, complex_contour_integral
from chebpy import cheb_interpolation_1d
from chebpy import clencurt_weights_fft, cheb_quadrature_clencurt
from chebpy import cheb_D1_mat
from chebpy import cheb_interpolation_1d

def test_cheb_mde_dirichlet():
    L = 10
    N = 64
    Ns = 101

    ii = np.arange(N+1)
    x = 1. * ii * L / N
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech
    plt.plot(x, W)
    plt.axis([0, 10, -1.1, 1.1])
    plt.show()

    u0 = np.ones_like(x)
    u0[0] = 0.; u0[N] = 0.;
    q1, x1 = cheb_mde_oss(W, u0, L, Ns)

    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech
    plt.plot(x, W)
    plt.axis([0, 10, -1.1, 1.1,])
    plt.show()

    Ns = 11
    u0 = np.ones((N+1, 1))
    u0[0] = 0; u0[N] = 0;
    q2, x2 = cheb_mde_etdrk4(W, u0, L, Ns)

    plt.plot(x1, q1)
    #plt.plot(x1, q1, '.')
    plt.plot(x2, q2, 'r')
    #plt.plot(x2, q2, 'r.')
    plt.axis([0, 10, 0, 1.15])
    plt.show()


def test_cheb_mde_neumann():
    L = 10
    N = 256
    Ns = 201

    ii = np.arange(N+1)
    x = 1. * ii * L / N
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech
    plt.plot(x, W)
    plt.axis([0, 10, -1.1, 1.1,])
    plt.show()

    u0 = np.ones_like(W)
    #q1, x1 = cheb_mde_neumann_split(W, u0, L, Ns)
    q1, x1 = cheb_mde_osc(W, u0, L, Ns)

    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech
    plt.plot(x, W)
    plt.axis([0, 10, -1.1, 1.1,])
    plt.show()

    u0 = np.ones_like(W)
    q2, x2 = cheb_mde_neumann_etdrk4(W, u0, L, Ns)
    q3, x3 = cheb_mde_robin_etdrk4(W, u0, L, Ns, 0., 0.)
    print np.max(np.abs(q2-q3))

    plt.plot(x1, q1, 'b')
    #plt.plot(x1, q1, 'b.')
    plt.plot(x2, q2, 'r')
    #plt.plot(x2, q2, 'r.')
    plt.plot(x3, q3, 'g')
    plt.axis([0, 10, 0, 1.15])
    plt.show()


def test_cheb_mde_robin():
    L = 10
    N = 128
    ka = 1.
    kb = -1.
    Ns = 101

    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech
    plt.plot(x, W)
    plt.axis([0, 10, -1.1, 1.1,])
    plt.show()

    u0 = np.ones_like(W)
    q1, x1 = cheb_mde_robin_etdrk4_1(W, u0, L, Ns, 0., 0.)
    #q2, x2 = cheb_mde_neumann_etdrk4(W, u0, L, Ns)
    q2, x2 = cheb_mde_robin_etdrk4_1(W, u0, L, Ns, 10*L*ka, .1*L*kb)
    q3, x3 = cheb_mde_robin_etdrk4_3(W, u0, L, Ns, 10*L*ka, .1*L*kb)
    #q3, x3 = cheb_mde_robin_etdrk4_2(W, u0, L, Ns, 0., 0.)

    plt.plot(x1, q1, 'b')
    plt.plot(x2, q2, 'r')
    plt.plot(x3, q3, 'g')
    plt.axis([0, 10, 0, 1.15])
    plt.show()


def test_cheb_mde_mixed():
    L = 10
    N = 64
    ka = 1.
    kb = -1.
    Ns = 1001

    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech
    plt.plot(x, W)
    plt.axis([0, 10, -1.1, 1.1,])
    plt.show()

    u0 = np.ones_like(W)
    u0[0] = 0.
    q1, x1 = cheb_mde_mixed_etdrk4(W, u0, L, Ns, 0*ka*L)

    plt.plot(x1, q1, 'b')
    plt.axis([0, 10, 0, 1.15])
    plt.show()


def test_accuracy_cheb_mde_dirichlet():
    L = 10
    N = 64
    Ns_ref = 20000+1 # highest accuracy for reference. h = 1e-4

    ii = np.arange(N+1)
    x = 1. * ii * L / N
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W1 = 1. - 2. * sech * sech

    u0 = np.ones_like(W1)
    u0[0] = 0.; u0[N] = 0.
    q1_ref, x1 = cheb_mde_oss(W1, u0, L, Ns_ref)

    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W2 = 1. - 2. * sech * sech

    u0 = np.ones_like(W2)
    u0[0] = 0.; u0[N] = 0.
    q2_ref, x2 = cheb_mde_etdrk4(W2, u0, L, Ns_ref)

    q1_ref2 = cheb_interpolation_1d(np.linspace(-1,1,N+1), q2_ref)
    q1_ref2.shape = (N+1,)
    print np.max(np.abs(q1_ref2 - q1_ref)) / np.max(q1_ref)
    print np.linalg.norm(q1_ref2 - q1_ref) / N

    plt.plot(x1, q1_ref)
    plt.plot(x1, q1_ref2)
    plt.plot(x2, q2_ref, 'r')
    plt.show()

    # Ns = 10^t
    errs1 = []
    errs2 = []
    Nss = []
    for Ns in np.round(np.power(10, np.linspace(0,4,10))):
        Ns = int(Ns) + 1
        u0 = np.ones_like(W1)
        u0[0] = 0.; u0[N] = 0.
        q1, x1 = cheb_mde_oss(W1, u0, L, Ns)
        u0 = np.ones_like(W2)
        u0[0] = 0.; u0[N] = 0.
        q2, x2 = cheb_mde_etdrk4(W2, u0, L, Ns)
        err1 = np.max(np.abs(q1 - q1_ref)) / np.max(q1_ref)
        err2 = np.max(np.abs(q2 - q2_ref)) / np.max(q2_ref)
        #err1 = np.linalg.norm(q1-q1_ref) / N
        #err2 = np.linalg.norm(q2-q2_ref) / N
        Nss.append(1./Ns)
        errs1.append(err1)
        errs2.append(err2)
        print Ns, '\t\t', err1, '\t\t\t', err2

    plt.plot(Nss, errs1)
    plt.plot(Nss, errs1, '.')
    plt.plot(Nss, errs2, 'r')
    plt.plot(Nss, errs2, 'r.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative timestep')
    plt.ylabel('Relative Error at s=1')
    plt.grid('on')
    plt.show()


def test_accuracy_cheb_mde_neumann():
    L = 10
    N = 64
    Ns_ref = 20000+1 # highest accuracy for reference. h = 1e-4

    ii = np.arange(N+1)
    x = 1. * ii * L / N
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W1 = 1. - 2. * sech * sech

    u0 = np.ones_like(W1)
    #q1_ref, x1 = cheb_mde_neumann_split(W1, u0, L, Ns_ref)
    q1_ref, x1 = cheb_mde_osc(W1, u0, L, Ns_ref)

    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W2 = 1. - 2. * sech * sech

    u0 = np.ones_like(W2)
    q2_ref, x2 = cheb_mde_neumann_etdrk4(W2, u0, L, Ns_ref)
    q3_ref, x3 = cheb_mde_robin_etdrk4(W2, u0, L, Ns_ref, 0., 0.)

    plt.plot(x1, q1_ref)
    plt.plot(x2, q2_ref, 'r')
    plt.plot(x3, q3_ref, 'g')
    plt.plot(x4, q4_ref, 'k')
    plt.show()

    # Ns = 10^t
    errs1 = []
    errs2 = []
    errs3 = []
    Nss = []
    for Ns in np.round(np.power(10, np.linspace(0,4,10))):
        Ns = int(Ns) + 1
        q1, x1 = cheb_mde_osc(W1, u0, L, Ns)
        q2, x2 = cheb_mde_neumann_etdrk4(W2, u0, L, Ns)
        q3, x3 = cheb_mde_robin_etdrk4(W2, u0, L, Ns, 0., 0.)
        #err1 = np.max(q1 - q1_ref) / np.max(q1_ref)
        #err2 = np.max(q2 - q2_ref) / np.max(q2_ref)
        #err3 = np.max(q3 - q3_ref) / np.max(q3_ref)
        err1 = np.linalg.norm(q1-q1_ref) / N
        err2 = np.linalg.norm(q2-q2_ref) / N
        err3 = np.linalg.norm(q3-q3_ref) / N
        Nss.append(1./Ns)
        errs1.append(err1)
        errs2.append(err2)
        errs3.append(err3)

    plt.plot(Nss, errs1, 'b')
    plt.plot(Nss, errs1, 'b.')
    plt.plot(Nss, errs2, 'r')
    plt.plot(Nss, errs2, 'r.')
    plt.plot(Nss, errs3, 'g')
    plt.plot(Nss, errs3, 'g.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative timestep')
    plt.ylabel('Relative Error at s=1')
    plt.grid('on')
    plt.show()


def test_accuracy_cheb_mde_robin():
    L = 10
    N = 64
    ka = .1
    kb = -1.
    Ns_ref = 20000+1 # highest accuracy for reference. h = 1e-4

    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W3 = 1. - 2. * sech * sech

    q3_ref, x3 = cheb_mde_robin_etdrk4(W3, L, Ns_ref, .5*L*ka, .5*L*kb)

    plt.plot(x3, q3_ref, 'g')
    plt.show()

    # Ns = 10^t
    errs3 = []
    Nss = []
    for Ns in np.round(np.power(10, np.linspace(0,4,10))):
        Ns = int(Ns) + 1
        q3, x3 = cheb_mde_robin_etdrk4(W3, L, Ns, .5*L*ka, .5*L*kb)
        err3 = np.max(q3 - q3_ref) / np.max(q3_ref)
        Nss.append(1./Ns)
        errs3.append(err3)

    plt.plot(Nss, errs3, 'g')
    plt.plot(Nss, errs3, 'g.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative timestep')
    plt.ylabel('Relative Error at s=1')
    plt.grid('on')
    plt.show()


def test_speed_space_split():
    '''
    The complexity O(NlnN) is confirmed.
    '''
    L = 10
    n = 18 # Nmax = 2^n
    Ns = 200+1 # highest accuracy for reference. h = 1e-4
    M_array = np.ones(n-1) # number of same run
    M_array[0:5] = 2**np.arange(5,0,-1)

    N_array = []
    t_array = []
    i = 0
    for N in 2**np.arange(2, n+1):
        ii = np.arange(N+1)
        x = 1. * ii * L / N
        sech = 1. / np.cosh(.75 * (2.*x - L))
        W = 1. - 2. * sech * sech
        
        t = time()
        for m in xrange(int(M_array[i])):
            q, x = cheb_mde_split(W, L, Ns)
        t_array.append((time()-t)/M_array[i])
        print N, '\t', t_array[-1]
        N_array.append(N)
        i += 1

    plt.plot(N_array, t_array)
    plt.plot(N_array, t_array, '.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Computer time')
    plt.grid('on')
    plt.show()


def test_speed_space_etdrk4():
    '''
    The expect complexity is O(N^2)
    However, due to the calculation of matrix exponential,
    it exceeds O(N^2) for large N.
    '''
    L = 10
    n = 13 # Nmax = 2^n
    Ns = 200+1 # highest accuracy for reference. h = 1e-4
    M_array = np.ones(n-1) # number of same run
    M_array[0:5] = 2**np.arange(5,0,-1)

    N_array = []
    t_array = []
    i = 0
    for N in 2**np.arange(2, n+1):
        ii = np.arange(N+1)
        x = np.cos(np.pi * ii / N)
        x = .5 * (x + 1) * L
        sech = 1. / np.cosh(.75 * (2.*x - L))
        W = 1. - 2. * sech * sech
        
        t = time()
        u0 = np.ones_like(x)
        for m in xrange(int(M_array[i])):
            q, x = cheb_mde_etdrk4(W, u0, L, Ns)
        t_array.append((time()-t)/M_array[i])
        print N, '\t', t_array[-1]
        N_array.append(N)
        i += 1

    plt.plot(N_array, t_array)
    plt.plot(N_array, t_array, '.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Computer time')
    plt.grid('on')
    plt.show()


def test_speed_accuracy_dirichlet():
    L = 10
    N = 64

    # Construct reference solution
    Ns_ref2 = 20000+1 # highest accuracy, h = 5e-5
    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W2 = 1. - 2. * sech * sech
    u0 = np.ones_like(x)
    u0[0] = 0.; u0[N] = 0.;
    q2_ref, x2 = cheb_mde_etdrk4(W2, u0, L, Ns_ref2)
    kk = np.arange(N+1)
    yy = (2. / N) * kk - 1.
    y = .5 * L * (yy + 1)
    q1_ref = cheb_interpolation_1d(yy, q2_ref)

    print 'Test etdrk4'
    # Ns = 10^t
    errs2 = []
    ts2 = []
    Nss2 = []
    for Ns in np.round(np.power(10, np.linspace(0,4,10))):
        Ns = int(Ns) + 1
        u0 = np.ones_like(x2)
        u0[0] = 0.; u0[N] = 0.;
        t = time()
        q2, x2 = cheb_mde_etdrk4(W2, u0, L, Ns)
        t = time() - t
        #err2 = np.max(q2 - q2_ref) / np.max(q2_ref)
        err2 = np.linalg.norm(q2-q2_ref) / N
        Nss2.append(1./Ns)
        errs2.append(err2)
        ts2.append(t)
        print Ns, '\t\t', err2, '\t\t', t

    print 'Test splitting'
    Ns_ref1 = 2000000+1 # highest accuracy for reference. h = 5e-7
    ii = np.arange(N+1)
    x = 1. * ii * L / N
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W1 = 1. - 2. * sech * sech
    u0 = np.ones_like(x)
    u0[0] = 0.; u0[N] = 0.;
    q1_ref0, x1 = cheb_mde_oss(W1, u0, L, Ns_ref1)
    plt.plot(x2, q2_ref)
    plt.plot(x1, q1_ref, 'r.')
    plt.plot(x1, q1_ref0, 'r')
    plt.show()
    # Ns = 10^t
    errs1 = []
    ts1 = [] # computer time
    Nss1 = []
    for Ns in np.round(np.power(10, np.linspace(0,6,15))):
        Ns = int(Ns) + 1
        u0 = np.ones_like(x)
        u0[0] = 0.; u0[N] = 0.;
        t = time()
        q1, x1 = cheb_mde_oss(W1, u0, L, Ns)
        t = time() - t
        #err1 = np.max(q1 - q1_ref) / np.max(q1_ref)
        err1 = np.linalg.norm(q1-q1_ref0) / N
        Nss1.append(1./Ns)
        errs1.append(err1)
        ts1.append(t)
        print Ns, '\t\t', err1, '\t\t', t

    plt.plot(Nss1, errs1)
    plt.plot(Nss1, errs1, '.')
    plt.plot(Nss2, errs2, 'r')
    plt.plot(Nss2, errs2, 'r.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative timestep')
    plt.ylabel('Relative Error at s=1')
    plt.grid('on')
    plt.show()

    plt.plot(errs1, ts1)
    plt.plot(errs1, ts1, '.')
    plt.plot(errs2, ts2, 'r')
    plt.plot(errs2, ts2, 'r.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative Error at s=1')
    plt.ylabel('Computer Time')
    plt.grid('on')
    plt.show()


def test_cheb_allen_cahn_etdrk4():
    cheb_allen_cahn_etdrk4()


def test_cheb_mde_brush():
    '''
    Solving MDE for polymer brushes.
    The Dirac initial condition is approximate by a Kronecker delta.

    Suppose the Dirac is \delta(x-x0), Kronecker delta is d_x0

    For Splitting method, the space is uniformly discretized, to constrain
    the integral of Dirac function to 1, that is
        I = \Integrate d_x0 = q[x0] * (L/N) = 1
    Thus, the initial condition of q is
        q(x) = N/L, x = x0
        q(x) = 0,   otherwise.

    For ETDRK4 method, the space is discretized in a manner that grids 
    are clustered near the boundary (Chebyshev Gauss-Lobatto Grids). We
    can evaluate the integral by Clenshaw-Curtis quadrature, that is
        I = \Integrate d_x0 = (L/2) * q[x0] * w[x0] = 1
    where w[x0] is the Clenshaw-Curtis weight at x0. Thus the initial
    condition of q is
        q(x) = 2/(L*w(x)), x = x0
        q(x) = 0,          otherwise

    Apporximate Dirac delta via Chebyshev differention of a step function,
        H_x0 = 0, x < x0
        H_x0 = 1, x >= x0
    Then the Driac delta is
        D .dot. H_x0
    Where D is Chebyshev first order differentiation matrix.
    Ref: Jung Jae-Hun, A spectral collocation approximation of one
        dimensional head-on colisions of black-holes.

    '''

    L = 15.0
    x0 = 0.2
    N = 256
    Ns = 200 + 1
    ds = 1./(Ns-1)

    ii = np.arange(N+1)
    x = 1. * ii * L / N
    W = np.zeros_like(x)
    u0 = np.zeros(N+1)
    ix = int(N * x0 / L)
    print ix, x[ix]
    u0[ix] = N / L

    q1, x1 = cheb_mde_split(W, u0, L, Ns)
    print np.abs(np.max(q1) - 1./np.sqrt(4.*np.pi*ds))

    N = 256
    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    W = np.zeros_like(x)
    # Apporximate Dirac delta via Kronecker delta
    u0 = np.zeros(N+1)
    w = clencurt_weights_fft(N)
    ix = int(np.arccos(2*x0/L-1) / np.pi * N)
    print ix, x[ix]
    u0[ix] = (2.0/L) / w[ix]
    # Apporximate Dirac delta via Chebyshev differentiation
    #D, xc = cheb_D1_mat(N)
    #H = np.zeros(N+1)
    #H[0:ix+1] = 1.
    #plt.plot(x, H)
    #plt.show()
    #u0 = np.dot(D, H)
    #u0 = u0 / cheb_quadrature_clencurt(u0)
    plt.plot(x, u0)
    plt.show()
    
    q2, x2 = cheb_mde_etdrk4(W, u0, L, Ns)
    print np.abs(np.max(q2) - 1./np.sqrt(4.*np.pi*ds))

    #plt.plot(x1, np.abs(q1))
    plt.plot(x1, np.abs(q1), '.')
    #plt.plot(x2, np.abs(q2), 'r')
    plt.plot(x2, np.abs(q2), 'r.')
    #plt.axis([0, 10, 0, 1.15])
    plt.yscale('log')
    plt.xlabel('x')
    plt.ylabel('q(x,ds)')
    plt.grid('on')
    plt.show()


def test_complex_contour_integral():
    M = 32
    R = 2.0
    for z in np.power(10, np.linspace(1,-9,11)):
        print z,'\t\t', complex_contour_integral(f, z, M, R)


def f(t):
    '''
    z       exact f(z)
    '''
    return (-4 - 3*t - t**2 + np.exp(t) * (4 - t)) / t**3


if __name__ == '__main__':
    #test_cheb_mde_dirichlet()
    #test_cheb_mde_neumann()
    test_cheb_mde_robin()
    #test_cheb_mde_mixed()
    #test_accuracy_cheb_mde_dirichlet()
    #test_accuracy_cheb_mde_neumann()
    #test_accuracy_cheb_mde_robin()
    #test_cheb_allen_cahn_etdrk4()
    #test_complex_contour_integral()
    #test_speed_space_etdrk4()
    #test_speed_space_split()
    #test_speed_accuracy_dirichlet()
    #test_cheb_mde_brush()

