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
from scipy.io import savemat, loadmat
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
import mpltex.acs # ACS configured matplotlib

from chebpy import cheb_mde_oss, cheb_mde_osc
from chebpy import cheb_mde_dirichlet_oscheb, cheb_mde_neumann_oscheb
from chebpy import cheb_mde_neumann_split
from chebpy import cheb_mde_dirichlet_etdrk4
from chebpy import cheb_mde_neumann_etdrk4
from chebpy import cheb_mde_robin_etdrk4, cheb_mde_robin_etdrk4_1
from chebpy import cheb_mde_robin_etdrk4_2
from chebpy import cheb_mde_mixed_etdrk4
from chebpy import cheb_allen_cahn_etdrk4, complex_contour_integral
from chebpy import cheb_interpolation_1d
from chebpy import clencurt_weights_fft, cheb_quadrature_clencurt
from chebpy import cheb_D1_mat, cheb_D2_mat_dirichlet_robin
from chebpy import cheb_D2_mat_dirichlet_dirichlet
from chebpy import cheb_D2_mat
from chebpy import cheb_interpolation_1d
from chebpy import cheb_quadrature_clencurt, oss_integral_weights
from chebpy import etdrk4_coeff_nondiag

def init_fourier(N, L, show=False):
    '''
    For equispaced grid.
    '''
    ii = np.arange(N+1)
    x = 1. * ii * L / N

    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech

    u0 = np.ones_like(x)

    if show:
        plt.figure()
        plt.plot(x, W, 'b')
        plt.axis([0, 10, -1.1, 1.1,])
        plt.xlabel('$z$')
        plt.ylabel('$w(z)$')
        plt.savefig('benchmark/w(z)', bbox_inches='tight')
        plt.show()

    return W, u0, x


def init_chebyshev(N, L, show=False):
    '''
    For Chebyshev grid.
    '''
    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L

    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech

    u0 = np.ones_like(x)

    if show:
        plt.figure()
        plt.plot(x, W, 'b')
        plt.axis([0, 10, -1.1, 1.1,])
        plt.xlabel('$z$')
        plt.ylabel('$w(z)$')
        plt.savefig('benchmark/w(z)', bbox_inches='tight')
        plt.show()

    return W, u0, x


def test_cheb_mde_dirichlet():
    L = 10
    Ns = 200 + 1

    print 'test_cheb_mde_dirichlet'

    N = 32
    W, u0, x = init_fourier(N, L)
    u0[0] = 0.; u0[N] = 0.;
    print 'OSS N = ', N, ' Ns= ', Ns-1
    #plt.plot(x, W)
    #plt.axis([0, 10, -1.1, 1.1])
    #plt.show()
    q1, x1 = cheb_mde_oss(W, u0, L, Ns)
    Q1 = L * oss_integral_weights(q1)

    N = 32
    W, u0, x2 = init_chebyshev(N, L)
    u0[0] = 0; u0[N] = 0;
    print 'OSCHEB N = ', N, ' Ns= ', Ns-1
    #plt.plot(x, W)
    #plt.axis([0, 10, -1.1, 1.1,])
    #plt.show()
    q2 = cheb_mde_dirichlet_oscheb(W, u0, L, Ns)
    Q2 = 0.5 * L * cheb_quadrature_clencurt(q2)

    N = 32
    Ns = 200 + 1
    algo = 1
    scheme = 1
    W, u0, x = init_chebyshev(N, L)
    u0[0] = 0; u0[N] = 0;
    print 'ETDRK4 N = ', N, ' Ns= ', Ns-1
    q3, x3 = cheb_mde_dirichlet_etdrk4(W, u0, L, Ns, algo, scheme)
    Q3 = 0.5 * L * cheb_quadrature_clencurt(q3)

    plt.plot(x1, q1, 'b')
    plt.plot(x2, q2, 'g')
    plt.plot(x3, q3, 'r')
    plt.axis([0, 10, 0, 1.15])
    plt.show()


def test_exact_dirichlet(oss=0,oscheb=0,etdrk4=0):
    L = 10

    if oss:
        print 'test_cheb_mde_dirichlet'
        N = 4096
        Ns = 100000 + 1
        W, u0, x = init_fourier(N, L)
        u0[0] = 0.; u0[N] = 0.;
        print 'OSS N = ', N, ' Ns = ', Ns-1
        q1, x1 = cheb_mde_oss(W, u0, L, Ns)
        Q1 = L * oss_integral_weights(q1)
        data_name = 'benchmark/exact/OSS_N' + str(N) + '_Ns' + str(Ns-1)
        savemat(data_name,{
                'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
                'x':x, 'q':q1, 'Q':Q1})
        plt.plot(x1, q1, 'b')
        plt.axis([0, 10, 0, 1.15])
        plt.show()

    if oscheb:
        N = 16384
        Ns = 1000000 + 1
        W, u0, x = init_chebyshev(N, L)
        u0[0] = 0; u0[N] = 0;
        print 'OSCHEB N = ', N, ' Ns = ', Ns-1
        q2 = cheb_mde_dirichlet_oscheb(W, u0, L, Ns)
        Q2 = 0.5 * L * cheb_quadrature_clencurt(q2)
        data_name = 'benchmark/exact/OSCHEB_N' + str(N) + '_Ns' + str(Ns-1)
        savemat(data_name,{
                'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
                'x':x, 'q':q2, 'Q':Q2})
        plt.plot(x, q2, 'g')
        plt.axis([0, 10, 0, 1.15])
        plt.xlabel('$z$')
        plt.ylabel('$q(z)$')
        plt.savefig(data_name, bbox_inches='tight')
        plt.show()

    if etdrk4:
        N = 128
        Ns = 20000 + 1
        algo = 1
        scheme = 1
        W, u0, x = init_chebyshev(N, L)
        u0[0] = 0; u0[N] = 0;
        print 'ETDRK4 N = ', N, ' Ns = ', Ns-1
        q3, x3 = cheb_mde_dirichlet_etdrk4(W, u0, L, Ns, algo, scheme)
        Q3 = 0.5 * L * cheb_quadrature_clencurt(q3)
        data_name = 'benchmark/exact/ETDRK4_N' + str(N) + '_Ns' + str(Ns-1)
        savemat(data_name,{
                'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
                'x':x, 'q':q3, 'Q':Q3})
        plt.plot(x3, q3, 'r')
        plt.axis([0, 10, 0, 1.15])
        plt.xlabel('$z$')
        plt.ylabel('$q(z)$')
        plt.savefig(data_name, bbox_inches='tight')
        plt.show()


def test_exact_neumann(oss=0,oscheb=0,etdrk4=0):
    L = 10

    if oss:
        print 'test_cheb_mde_dirichlet'
        N = 4096
        Ns = 100000 + 1
        W, u0, x = init_fourier(N, L)
        u0[0] = 0.; u0[N] = 0.;
        print 'OSS N = ', N, ' Ns = ', Ns-1
        q1, x1 = cheb_mde_oss(W, u0, L, Ns)
        Q1 = L * oss_integral_weights(q1)
        data_name = 'benchmark/exact/OSS_N' + str(N) + '_Ns' + str(Ns-1)
        savemat(data_name,{
                'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
                'x':x, 'q':q1, 'Q':Q1})
        plt.plot(x1, q1, 'b')
        plt.axis([0, 10, 0, 1.15])
        plt.show()

    if oscheb:
        N = 16384
        Ns = 1000000 + 1
        W, u0, x = init_chebyshev(N, L)
        u0[0] = 0; u0[N] = 0;
        print 'OSCHEB N = ', N, ' Ns = ', Ns-1
        q2 = cheb_mde_dirichlet_oscheb(W, u0, L, Ns)
        Q2 = 0.5 * L * cheb_quadrature_clencurt(q2)
        data_name = 'benchmark/exact/OSCHEB_N' + str(N) + '_Ns' + str(Ns-1)
        savemat(data_name,{
                'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
                'x':x, 'q':q2, 'Q':Q2})
        plt.plot(x, q2, 'g')
        plt.axis([0, 10, 0, 1.15])
        plt.xlabel('$z$')
        plt.ylabel('$q(z)$')
        plt.savefig(data_name, bbox_inches='tight')
        plt.show()

    if etdrk4:
        N = 128
        Ns = 20000 + 1
        algo = 1
        scheme = 1
        W, u0, x = init_chebyshev(N, L)
        u0[0] = 0; u0[N] = 0;
        print 'ETDRK4 N = ', N, ' Ns = ', Ns-1
        q3, x3 = cheb_mde_dirichlet_etdrk4(W, u0, L, Ns, algo, scheme)
        Q3 = 0.5 * L * cheb_quadrature_clencurt(q3)
        data_name = 'benchmark/exact/ETDRK4_N' + str(N) + '_Ns' + str(Ns-1)
        savemat(data_name,{
                'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
                'x':x, 'q':q3, 'Q':Q3})
        plt.plot(x3, q3, 'r')
        plt.axis([0, 10, 0, 1.15])
        plt.xlabel('$z$')
        plt.ylabel('$q(z)$')
        plt.savefig(data_name, bbox_inches='tight')
        plt.show()


def test_cheb_mde_neumann():
    L = 10
    N = 128
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
    q4, x4 = cheb_mde_robin_etdrk4_2(W, u0, L, Ns, 0., 0.)
    q5 = cheb_mde_neumann_oscheb(W, u0, L, Ns)
    print np.max(np.abs(q2-q3))
    print np.max(np.abs(q4-q3))

    plt.plot(x1, q1, 'b')
    #plt.plot(x1, q1, 'b.')
    plt.plot(x2, q2, 'r')
    #plt.plot(x2, q2, 'r.')
    plt.plot(x3, q3, 'g')
    plt.plot(x4, q4, 'k')
    plt.plot(x4, q5, 'm')
    plt.axis([0, 10, 0, 1.15])
    plt.show()


def test_cheb_mde_robin():
    L = 10
    N = 128
    ka = -1.0
    kb = 0.1
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
    q0, x0 = cheb_mde_robin_etdrk4(W, u0, L, Ns, L*ka, L*kb)
    q1, x1 = cheb_mde_robin_etdrk4_1(W, u0, L, Ns, L*ka, L*kb)
    q2, x2 = cheb_mde_robin_etdrk4_2(W, u0, L, Ns, L*ka, L*kb)
    #q3, x3 = cheb_mde_robin_etdrk4_3(W, u0, L, Ns, L*ka, L*kb)

    plt.plot(x0, q0, 'k')
    plt.plot(x1, q1, 'b')
    plt.plot(x2, q2, 'r')
    #plt.plot(x3, q3, 'g')
    plt.axis([0, 10, 0, 1.15])
    plt.show()


def test_cheb_mde_mixed():
    L = 10
    N = 128
    ka = -0.0
    kb = 0.0
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
    u0[-1] = 0.
    q1, x1 = cheb_mde_mixed_etdrk4(W, u0, L, Ns, kb*L)

    plt.plot(x1, q1, 'b')
    plt.axis([0, 10, 0, 1.15])
    plt.show()


def test_Ns_dirichlet():
    '''
    1 OSS
    2 OSCHEB
    3 ETDRK4
    '''
    L = 10
    N = 160
    Ns1 = 200000 + 1
    Ns2 = 200000 + 1
    Ns3 = 200000 + 1
    method = 2 # 0: OSS, 1: OSCHEB, 2: ETDRK4 
    algo3 = 1 # 0: circular, 1: hyperbolic, 2: scaling and squaring
    scheme = 1
    data_name = 'benchmark/Ns_convergence/'
    if method == 0:
        data_name = data_name + 'OSS/Ns_DBC_N' + str(N)
    elif method == 1:
        data_name = data_name + 'OSCHEB/Ns_DBC_N' + str(N)
    elif method == 2:
        if scheme == 0:
            data_name = data_name + 'Cox_Matthews/Ns_DBC_N' + str(N)
        else:
            data_name = data_name + 'Krogstad//Ns_DBC_N' + str(N)
    else:
        raise ValueError('No such method!')

    print data_name

    oscheb_ref = 'benchmark/exact/OSCHEB_N'
    oscheb_ref = oscheb_ref + '8192_Ns200000.mat'
    mat = loadmat(oscheb_ref)
    q2_ref = mat['q']
    Q2_ref = mat['Q'][0,0]
    N2_ref = mat['N']
    Ns2_ref = mat['Ns']

    #oss_ref = 'benchmark/exact/OSS_N'
    #oss_ref = oss_ref + str(N) + '_Ns20000.mat'
    oss_ref = oscheb_ref
    mat = loadmat(oss_ref)
    q1_ref = mat['q']
    Q1_ref = mat['Q'][0,0]
    N1_ref = mat['N']
    Ns1_ref = mat['Ns']

    #if scheme == 0:
    #    etdrk4_ref = 'benchmark/exact/CoxMatthews/ETDRK4_N'
    #elif scheme == 1:
    #    etdrk4_ref = 'benchmark/exact/Krogstad/ETDRK4_N'
    #else:
    #    raise ValueError('No such scheme!')
    #etdrk4_ref = etdrk4_ref + str(N) + '_Ns20000.mat'
    etdrk4_ref = oscheb_ref
    mat = loadmat(etdrk4_ref)
    q3_ref = mat['q']
    Q3_ref = mat['Q'][0,0]
    print Q3_ref.shape
    N3_ref = mat['N']
    Ns3_ref = mat['Ns']

    mode = 0 # error mode 0: Q only, 1: q and Q
    if N1_ref == N and N2_ref.size == N and N3_ref.size == N:
        mode = 1

    print 'OSS'
    if method == 0:
        iters = 10
    else:
        iters = 0
    errs0_1 = []
    errs1_1 = []
    Qs1 = []
    Nss1 = []
    W1, u0, x = init_fourier(N, L)
    u0[0] = 0.; u0[-1] = 0.
    ns_max = int(np.log10((Ns1-1)/2)) # Ns_max = 10^{ns_max}
    for Ns in np.power(10, np.linspace(0,ns_max,iters)).astype(int) + 1:
        q1, x1 = cheb_mde_oss(W1, u0, L, Ns)
        Q1 = L * oss_integral_weights(q1)
        Qs1.append(Q1)
        if mode:
            err1 = np.max(np.abs(q1 - q1_ref)) / np.max(q1_ref)
            errs0_1.append(err1)
        err1 = np.abs(Q1 - Q1_ref) / np.abs(Q1_ref)
        errs1_1.append(err1)
        Nss1.append(1./(Ns-1))
        print Ns-1, '\t', err1

    print 'OSCHEB'
    if method == 1:
        iters = 10
    else:
        iters = 0
    errs0_2 = []
    errs1_2 = []
    Qs2 = []
    Nss2 = []
    W2, u0, x = init_chebyshev(N, L)
    u0[0] = 0.; u0[-1] = 0.
    ns_max = int(np.log10((Ns2-1)/2)) # Ns_max = 10^{ns_max}
    for Ns in np.power(10, np.linspace(0,ns_max,iters)).astype(int) + 1:
        q2 = cheb_mde_dirichlet_oscheb(W2, u0, L, Ns)
        Q2 = 0.5 * L * cheb_quadrature_clencurt(q2)
        Qs2.append(Q2)
        if mode:
            err2 = np.max(np.abs(q2 - q2_ref)) / np.max(q2_ref)
            errs0_2.append(err2)
        err2 = np.abs(Q2 - Q2_ref) / np.abs(Q2_ref)
        errs1_2.append(err2)
        Nss2.append(1./(Ns-1))
        print Ns-1, '\t', err2

    if scheme == 0:
        print 'ETDRK4-Cox-Matthews'
    else:
        print 'ETDRK4-Krogstad'
    if method == 2:
        iters = 10
    else:
        iters = 0
    errs0_3 = []
    errs1_3 = []
    Qs3 = []
    Nss3 = []
    W3, u0, x = init_chebyshev(N, L)
    u0[0] = 0.; u0[-1] = 0.
    ns_max = int(np.log10((Ns3-1)/2)) # Ns_max = 10^{ns_max}
    for Ns in np.power(10, np.linspace(0,ns_max,iters)).astype(int) + 1:
        q3, x3 = cheb_mde_dirichlet_etdrk4(W3, u0, L, Ns, algo3, scheme)
        Q3 = 0.5 * L * cheb_quadrature_clencurt(q3)
        Qs3.append(Q3)
        if mode:
            err3 = np.max(np.abs(q3 - q3_ref)) / np.max(q3_ref)
            errs0_3.append(err3)
        err3 = np.abs(Q3 - Q3_ref) / np.abs(Q3_ref)
        errs1_3.append(err3)
        Nss3.append(1./(Ns-1))
        print Ns-1, '\t', err3

    savemat(data_name, {'N_ref1':N1_ref, 'N_ref2':N2_ref,
                        'N_ref3':N3_ref, 'N':N,
                        'Ns1_ref':Ns1_ref, 'Ns2_ref':Ns2_ref,
                        'Ns3_ref':Ns3_ref, 'Algorithm1':'OSS',
                        'Algorithm2':'OSCHEB', 'Algorithm3':'ETDRK4',
                        'Q1_ref':Q1_ref, 'Q2_ref':Q2_ref, 'Q3_ref':Q3_ref,
                        'Q1':Qs1, 'Q2':Qs2, 'Q3':Qs3,
                        'Ns0_1':Nss1, 'err0_1':errs0_1,
                        'Ns1_1':Nss1, 'err1_1':errs1_1,
                        'Ns0_2':Nss2, 'err0_2':errs0_2,
                        'Ns1_2':Nss2, 'err1_2':errs1_2,
                        'Ns0_3':Nss3, 'err0_3':errs0_3,
                        'Ns1_3':Nss3, 'err1_3':errs1_3})

    if mode:
        plt.plot(Nss1, errs0_1, 'bo-', label='OSS')
        plt.plot(Nss2, errs0_2, 'go-', label='OSCHEB')
        plt.plot(Nss3, errs0_3, 'ro-', label='ETDRK4')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$\Delta s$')
        plt.ylabel('Relative Error at s=1')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='lower right')
        plt.grid('on')
        plt.show()

    plt.plot(Nss1, errs1_1, 'bo-', label='OSS')
    plt.plot(Nss2, errs1_2, 'go-', label='OSCHEB')
    plt.plot(Nss3, errs1_3, 'ro-', label='ETDRK4')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\Delta s$')
    plt.ylabel('Relative Error at s=1')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')
    plt.grid('on')
    plt.savefig(data_name, bbox_inches='tight')
    plt.show()


def test_N_dirichlet():
    '''
    1 OSS
    2 OSCHEB
    3 ETDRK4
    '''
    err_mode = 1 # 0: max(|q-q_ref|)/max(q_ref); 1: |Q - Q_ref|/|Q_ref|
    L = 10
    N1 = 2
    N2 = 2048
    N3 = 2
    Ns1 = 100000 + 1
    Ns2 = 100000 + 1
    Ns3 = 100000 + 1
    algo3 = 1 # 0: circular, 1: hyperbolic, 2: scaling and squaring
    scheme = 1 # 0: Cox-Matthews, 2: Krogstad
    data_name = 'benchmark/N_DBC_Ns'+str(Ns1-1) + '_hyperbolic'
    print data_name

    oscheb_ref = 'benchmark/exact/OSCHEB_N'
    oscheb_ref = oscheb_ref + '8192_Ns200000.mat'
    mat = loadmat(oscheb_ref)
    q2_ref = mat['q']
    Q2_ref = mat['Q'][0,0]
    N2_ref = mat['N']
    Ns2_ref = mat['Ns']

    #oss_ref = 'benchmark/exact/OSS_N'
    #oss_ref = oss_ref + '4096_Ns100000.mat'
    oss_ref = oscheb_ref
    mat = loadmat(oss_ref)
    q1_ref = mat['q']
    Q1_ref = mat['Q'][0,0]
    N1_ref = mat['N']
    Ns1_ref = mat['Ns']

    #if scheme == 0:
    #    etdrk4_ref = 'benchmark/exact/CoxMatthews/ETDRK4_N'
    #elif scheme == 1:
    #    etdrk4_ref = 'benchmark/exact/Krogstad/ETDRK4_N'
    #else:
    #    raise ValueError('No such scheme!')
    #etdrk4_ref = etdrk4_ref + '256_Ns100000.mat'
    etdrk4_ref = oscheb_ref
    mat = loadmat(etdrk4_ref)
    q3_ref = mat['q']
    Q3_ref = mat['Q'][0,0]
    N3_ref = mat['N']
    Ns3_ref = mat['Ns']

    print 'OSS'
    errs1 = []
    Qs1 = []
    NN1 = []
    n_max = int(np.log2(N1)) # N_max = 2^{n_max - 1}
    for N in np.power(2, np.arange(4,n_max)).astype(int):
        W, u0, x = init_fourier(N, L)
        u0[0] = 0.; u0[-1] = 0.
        q1, x1 = cheb_mde_oss(W, u0, L, Ns1)
        Q1 = L * oss_integral_weights(q1)

        err1 = np.abs(Q1 - Q1_ref) / np.abs(Q1_ref)
        NN1.append(N)
        Qs1.append(Q1)
        errs1.append(err1)
        print N, '\t', err1

    print 'OSCHEB'
    errs2 = []
    Qs2 = []
    NN2 = []
    n_max = int(np.log2(N2)) # N_max = 2^{n_max - 1}
    for N in np.power(2, np.arange(4,n_max)).astype(int):
        W, u0, x = init_chebyshev(N, L)
        u0[0] = 0.; u0[-1] = 0.
        q2 = cheb_mde_dirichlet_oscheb(W, u0, L, Ns2)
        Q2 = 0.5 * L * cheb_quadrature_clencurt(q2)

        err2 = np.abs(Q2 - Q2_ref) / np.abs(Q2_ref)
        NN2.append(N)
        Qs2.append(Q2)
        errs2.append(err2)
        print N, '\t', err2

    print 'ETDRK4'
    errs3 = []
    Qs3 = []
    NN3 = []
    n_max = int(np.log2(N3)) # N_max = 2^{n_max - 1}
    for N in np.power(2, np.linspace(4,7.6,0)).astype(int):
        N = int(N)

        W, u0, x = init_chebyshev(N, L)
        u0[0] = 0.; u0[-1] = 0.
        q3, x3 = cheb_mde_dirichlet_etdrk4(W, u0, L, Ns3, algo3, scheme)
        Q3 = 0.5 * L * cheb_quadrature_clencurt(q3)

        err3 = np.abs(Q3 - Q3_ref) / np.abs(Q3_ref)
        NN3.append(N)
        Qs3.append(Q3)
        errs3.append(err3)
        print N, '\t', err3

    savemat(data_name, {'N_ref1':N1_ref, 'N_ref2':N2_ref,
                        'N_ref3':N3_ref, 'Ns1':Ns1-1,
                        'Ns2':Ns2-1, 'Ns3':Ns3-1,
                        'Ns1_ref':Ns1_ref, 'Ns2_ref':Ns2_ref,
                        'Ns3_ref':Ns3_ref, 'Algorithm1':'OSS',
                        'Algorithm2':'OSCHEB', 'Algorithm3':'ETDRK4',
                        'Q1_ref':Q1_ref, 'Q2_ref':Q2_ref, 'Q3_ref':Q3_ref,
                        'Q1':Qs1, 'Q2':Qs2, 'Q3':Qs3,
                        'N1':NN1, 'err1':errs1,
                        'N2':NN2, 'err2':errs2,
                        'N3':NN3, 'err3':errs3})

    plt.plot(NN1, errs1, 'bo-', label='OSS')
    plt.plot(NN2, errs2, 'go-', label='OSCHEB')
    plt.plot(NN3, errs3, 'ro-', label='ETDRK4')
    plt.yscale('log')
    plt.xlabel('$N_z$')
    plt.ylabel('Relative Error at s=1')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right')
    plt.grid('on')
    plt.savefig(data_name, bbox_inches='tight')
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
    q4_ref = cheb_mde_neumann_oscheb(W2, u0, L, Ns_ref)

    plt.plot(x1, q1_ref)
    plt.plot(x2, q2_ref, 'r')
    plt.plot(x3, q3_ref, 'g')
    plt.plot(x2, q4_ref, 'm')
    plt.show()

    # Ns = 10^t
    errs1 = []
    errs2 = []
    errs3 = []
    errs4 = []
    Nss = []
    for Ns in np.round(np.power(10, np.linspace(0,4,10))):
        Ns = int(Ns) + 1
        q1, x1 = cheb_mde_osc(W1, u0, L, Ns)
        q2, x2 = cheb_mde_neumann_etdrk4(W2, u0, L, Ns)
        q3, x3 = cheb_mde_robin_etdrk4(W2, u0, L, Ns, 0., 0.)
        q4 = cheb_mde_neumann_oscheb(W2, u0, L, Ns)
        #err1 = np.max(q1 - q1_ref) / np.max(q1_ref)
        #err2 = np.max(q2 - q2_ref) / np.max(q2_ref)
        #err3 = np.max(q3 - q3_ref) / np.max(q3_ref)
        #err4 = np.max(q4 - q4_ref) / np.max(q4_ref)
        err1 = np.linalg.norm(q1-q1_ref) / N
        err2 = np.linalg.norm(q2-q2_ref) / N
        err3 = np.linalg.norm(q3-q3_ref) / N
        err4 = np.linalg.norm(q4-q4_ref) / N
        Nss.append(1./Ns)
        errs1.append(err1)
        errs2.append(err2)
        errs3.append(err3)
        errs4.append(err4)

    plt.plot(Nss, errs1, 'b')
    plt.plot(Nss, errs1, 'b.')
    plt.plot(Nss, errs2, 'r')
    plt.plot(Nss, errs2, 'r.')
    plt.plot(Nss, errs3, 'g')
    plt.plot(Nss, errs3, 'g.')
    plt.plot(Nss, errs4, 'm')
    plt.plot(Nss, errs4, 'm.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative timestep')
    plt.ylabel('Relative Error at s=1')
    plt.grid('on')
    plt.show()


def test_accuracy_cheb_mde_robin():
    L = 10
    N = 64
    ka = .0
    kb = .0
    Ns_ref = 200000+1 # highest accuracy for reference. h = 1e-4

    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech

    u0 = np.ones_like(W)
    q1_ref, x1 = cheb_mde_robin_etdrk4(W, u0, L, Ns_ref, L*ka, L*kb)
    q2_ref, x2 = cheb_mde_robin_etdrk4_1(W, u0, L, Ns_ref, L*ka, L*kb)
    q3_ref, x3 = cheb_mde_robin_etdrk4_2(W, u0, L, Ns_ref, L*ka, L*kb)

    plt.plot(x1, q1_ref, 'b')
    plt.plot(x2, q2_ref, 'r')
    plt.plot(x3, q3_ref, 'g')
    plt.show()

    # Ns = 10^t
    errs1 = []
    errs2 = []
    errs3 = []
    Nss = []
    for Ns in np.round(np.power(10, np.linspace(0,4,10))):
        Ns = int(Ns) + 1
        q1, x1 = cheb_mde_robin_etdrk4(W, u0, L, Ns, L*ka, L*kb)
        q2, x2 = cheb_mde_robin_etdrk4_1(W, u0, L, Ns, L*ka, L*kb)
        q3, x3 = cheb_mde_robin_etdrk4_2(W, u0, L, Ns, L*ka, L*kb)
        err1 = np.max(q1 - q1_ref) / np.max(q1_ref)
        err2 = np.max(q2 - q2_ref) / np.max(q2_ref)
        err3 = np.max(q3 - q3_ref) / np.max(q3_ref)
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
            q, x = cheb_mde_oss(W, L, Ns)
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
    q2_ref, x2 = cheb_mde_dirichlet_etdrk4(W2, u0, L, Ns_ref2)
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
        q2, x2 = cheb_mde_dirichlet_etdrk4(W2, u0, L, Ns)
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

    q1, x1 = cheb_mde_oss(W, u0, L, Ns)
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
    
    q2, x2 = cheb_mde_dirichlet_etdrk4(W, u0, L, Ns)
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


def test_etdrk4_coefficients():
    Lx = 10
    N = 256
    Nsr = 600000 + 1
    hr = 1. / (Nsr - 1)
    
    M = 32; R = 15.
    D1, D2, x = cheb_D2_mat_dirichlet_dirichlet(N)
    L = (4. / Lx**2) * D2
    print np.linalg.norm(L*hr, 1)
    exit()
    Qr, f1r, f2r, f3r = etdrk4_coeff_nondiag(L, hr, M, R)

    errs1 = []
    errs2 = []
    errs3 = []
    errs4 = []
    hs = []
    for Ns in np.power(10, np.arange(1, 6)):
        h = 1. / Ns
        Q, f1, f2, f3 = etdrk4_coeff_nondiag(L, h, M, R)
        err1 = np.max(np.abs(Q - Qr))
        err2 = np.max(np.abs(f1 - f1r))
        err3 = np.max(np.abs(f2 - f2r))
        err4 = np.max(np.abs(f3 - f3r))
        errs1.append(err1)
        errs2.append(err2)
        errs3.append(err3)
        errs4.append(err4)
        hs.append(h)
        print Ns, '\t', err1, '\t', err2, '\t', err3, '\t', err4

    plt.plot(hs, errs1, 'bo-', mew=0, label='Q')
    plt.plot(hs, errs2, 'go-', mew=0, label='f1')
    plt.plot(hs, errs3, 'ro-', mew=0, label='f2')
    plt.plot(hs, errs4, 'ko-', mew=0, label='f3')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\Delta s$')
    plt.ylabel('Error for coeifficients')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='lower right')
    plt.grid('on')
    plt.show()


def test_mde_eig():
    Lx = 10
    N = 256
    Ns = 600000 + 1
    h = 1. / (Ns - 1)
    #D1, D2, x = cheb_D2_mat(N)
    D1, D2, x = cheb_D2_mat_dirichlet_dirichlet(N)
    #D1, D2, x = cheb_D2_mat_dirichlet_robin(N, 1.0)
    L = (4. / Lx**2) * D2
    eigv = eigvals(L*h)
    print np.max(eigv), np.min(eigv)


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
    #test_N_dirichlet()
    test_Ns_dirichlet()
    #test_exact_dirichlet(0,0,1)

    #test_cheb_mde_dirichlet()
    #test_cheb_mde_neumann()
    #test_cheb_mde_robin()
    #test_cheb_mde_mixed()
    #test_accuracy_cheb_mde_neumann()
    #test_accuracy_cheb_mde_robin()
    #test_cheb_allen_cahn_etdrk4()
    #test_complex_contour_integral()
    #test_speed_space_etdrk4()
    #test_speed_space_split()
    #test_speed_accuracy_dirichlet()
    #test_cheb_mde_brush()
    #test_mde_eig()
    #test_etdrk4_coefficients()
