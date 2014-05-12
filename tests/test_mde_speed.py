# -*- coding: utf-8 -*-
#/usr/bin/env python
"""
test_mde_speed
==============

Speed of modiffied diffusion equation (MDE) solvers.

"""

from time import time, clock
import numpy as np
from numpy.fft import fft, ifft
from scipy.io import savemat, loadmat
from scipy.linalg import eigvals
from scipy.integrate import simps, romb
import matplotlib.pyplot as plt

from timer import Timer
import mpltex.acs # ACS configured matplotlib

from chebpy import cheb_mde_oss, cheb_mde_osc, OSS, OSC
from chebpy import OSCHEB
from chebpy import BC, ETDRK4
from chebpy import clencurt_weights_fft, cheb_quadrature_clencurt
from chebpy import cheb_D1_mat, cheb_D2_mat_dirichlet_robin
from chebpy import cheb_D2_mat_dirichlet_dirichlet
from chebpy import cheb_D2_mat
from chebpy import cheb_interpolation_1d
from chebpy import oss_integral_weights
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


def init_chebyshev_fredrikson(N, L, show=False):
    '''
    For Chebyshev grid.
    '''
    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L

    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech

    u0 = np.ones_like(x)
    u0[0] = 0.; u0[-1] = 0.

    if show:
        plt.figure()
        plt.plot(x, W, 'b')
        plt.axis([0, 10, -1.1, 1.1,])
        plt.xlabel('$z$')
        plt.ylabel('$w(z)$')
        plt.savefig('benchmark/w(z)', bbox_inches='tight')
        plt.show()

    return W, u0, x


def init_chebyshev(N, L, show=True):
    '''
    For Chebyshev grid.
    '''
    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L

    W = -.1 * (np.pi * x / 4)**2

    u0 = np.zeros(N+1)
    w = clencurt_weights_fft(N)
    ix = 40
    u0[ix] = (2.0/L) / w[ix]

    if show:
        plt.figure()
        plt.plot(x, W, 'b')
        #plt.axis([0, 10, -1.1, 1.1,])
        plt.xlabel('$z$')
        plt.ylabel('$w(z)$')
        #plt.savefig('benchmark/w(z)', bbox_inches='tight')
        plt.show()
        plt.plot(x, u0, 'r')
        #plt.axis([0, 10, -1.1, 1.1,])
        plt.xlabel('$z$')
        plt.ylabel('$u0(z)$')
        plt.show()

    return W, u0, x


def test_exact_dirichlet(oss=0,oscheb=0,etdrk4=0):
    L = 10.0

    if oss:
        N = 1024 #4096
        Ns = 1000 + 1 #100000 + 1
        W, u0, x = init_fourier(N, L)
        u0[0] = 0.; u0[N] = 0.;
        print 'OSS N = ', N, ' Ns = ', Ns-1
        #q1, x1 = cheb_mde_oss(W, u0, L, Ns)
        oss_solver = OSS(L, N, Ns)
        q1, x1 = oss_solver.solve(W, u0)
        Q1 = L * oss_integral_weights(q1)
        #data_name = 'benchmark/exact/OSS_N' + str(N) + '_Ns' + str(Ns-1)
        data_name = 'OSS_N' + str(N) + '_Ns' + str(Ns-1)
        savemat(data_name,{
                'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
                'x':x, 'q':q1, 'Q':Q1})
        plt.plot(x1, q1, 'b')
        plt.axis([0, 10, 0, 1.15])
        #plt.show()

    if oscheb:
        N = 128 #16384
        Ns = 200 + 1 #1000000 + 1
        W, u0, x = init_chebyshev_fredrikson(N, L)
        u0[0] = 0; u0[N] = 0;
        print 'OSCHEB N = ', N, ' Ns = ', Ns-1
        #q2 = cheb_mde_dirichlet_oscheb(W, u0, L, Ns)
        oscheb_sovler = OSCHEB(L, N, Ns)
        q2, x2 = oscheb_sovler.solve(W, u0)
        Q2 = 0.5 * L * cheb_quadrature_clencurt(q2)
        #data_name = 'benchmark/exact/OSCHEB_N' + str(N) + '_Ns' + str(Ns-1)
        data_name = 'OSCHEB_N' + str(N) + '_Ns' + str(Ns-1)
        savemat(data_name,{
                'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
                'x':x2, 'q':q2, 'Q':Q2})
        plt.plot(x2, q2, 'g')
        plt.axis([0, 10, 0, 1.15])
        plt.xlabel('$z$')
        plt.ylabel('$q(z)$')
        plt.savefig(data_name, bbox_inches='tight')
        #plt.show()

    if etdrk4:
        N = 128
        Ns = 200 + 1 #20000 + 1
        algo = 1
        scheme = 1
        W, u0, x = init_chebyshev_fredrikson(N, L)
        u0[0] = 0; u0[N] = 0;
        print 'ETDRK4 N = ', N, ' Ns = ', Ns-1
        #q3, x3 = cheb_mde_dirichlet_etdrk4(W, u0, L, Ns, algo, scheme)
        etdrk4_solver = ETDRK4(L, N, Ns)
        q3, x3 = etdrk4_solver.solve(W, u0)
        Q3 = 0.5 * L * cheb_quadrature_clencurt(q3)
        #data_name = 'benchmark/exact/ETDRK4_N' + str(N) + '_Ns' + str(Ns-1)
        data_name = 'ETDRK4_N' + str(N) + '_Ns' + str(Ns-1)
        savemat(data_name,{
                'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
                'x':x, 'q':q3, 'Q':Q3})
        plt.plot(x3, q3, 'r')
        plt.axis([0, 10, 0, 1.15])
        plt.xlabel('$z$')
        plt.ylabel('$q(z)$')
        plt.savefig(data_name, bbox_inches='tight')
        plt.show()


def test_exact_neumann(osc=0,oscheb=0,etdrk4=0):
    L = 10.0

    if osc:
        N = 128
        Ns = 1000 + 1 #20000 + 1
        W, u0, x = init_fourier(N, L)
        print 'OSC N = ', N, ' Ns = ', Ns-1
        #q1, x1 = cheb_mde_osc(W, u0, L, Ns)
        osc_solver = OSC(L, N, Ns)
        q1, x1 = osc_solver.solve(W, u0)
        Q1 = L * simps(q1, dx=1./N)
        #data_name = 'benchmark/NBC-NBC/exact/OSS_N' 
        data_name = 'OSS_N' 
        data_name = data_name + str(N) + '_Ns' + str(Ns-1)
        savemat(data_name,{
                'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
                'x':x, 'q':q1, 'Q':Q1})
        plt.plot(x1, q1, 'b')
        plt.axis([0, 10, 0, 1.15])
        plt.xlabel('$z$')
        plt.ylabel('$q(z)$')
        plt.savefig(data_name, bbox_inches='tight')
        #plt.show()

    if oscheb:
        N = 128
        Ns = 200 + 1 #20000 + 1
        W, u0, x = init_chebyshev_fredrikson(N, L)
        print 'OSCHEB N = ', N, ' Ns = ', Ns-1
        #q2 = cheb_mde_neumann_oscheb(W, u0, L, Ns)
        oscheb_sovler = OSCHEB(L, N, Ns, bc=BC('Neumann'))
        q2, x2 = oscheb_sovler.solve(W, u0)
        Q2 = 0.5 * L * cheb_quadrature_clencurt(q2)
        #data_name = 'benchmark/NBC-NBC/exact/OSCHEB_N'
        data_name = 'OSCHEB_N'
        data_name = data_name + str(N) + '_Ns' + str(Ns-1)
        savemat(data_name,{
                'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
                'x':x2, 'q':q2, 'Q':Q2})
        plt.plot(x2, q2, 'g')
        plt.axis([0, 10, 0, 1.15])
        plt.xlabel('$z$')
        plt.ylabel('$q(z)$')
        plt.savefig(data_name, bbox_inches='tight')
        #plt.show()

    if etdrk4:
        N = 128
        Ns = 200 + 1
        algo = 1
        scheme = 1
        W, u0, x = init_chebyshev_fredrikson(N, L)
        print 'ETDRK4 N = ', N, ' Ns = ', Ns-1
        #q3, x3 = cheb_mde_neumann_etdrk4(W, u0, L, Ns, None, algo, scheme)
        lbc = BC('Neumann')
        rbc = BC('Neumann')
        etdrk4_solver = ETDRK4(L, N, Ns, h=None, lbc=lbc, rbc=rbc)
        q3, x3 = etdrk4_solver.solve(W, u0)
        Q3 = 0.5 * L * cheb_quadrature_clencurt(q3)
        #if scheme == 0:
        #    data_name = 'benchmark/NBC-NBC/exact/ETDRK4_Cox_N' 
        #    data_name = data_name + str(N) + '_Ns' + str(Ns-1)
        #else:
        #    data_name = 'benchmark/NBC-NBC/exact/ETDRK4_Krogstad_N' 
        #    data_name = data_name + str(N) + '_Ns' + str(Ns-1)
        #savemat(data_name,{
        #        'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
        #        'x':x, 'q':q3, 'Q':Q3})
        plt.plot(x3, q3, 'r')
        plt.axis([0, 10, 0, 1.15])
        plt.xlabel('$z$')
        plt.ylabel('$q(z)$')
        #plt.savefig(data_name, bbox_inches='tight')
        plt.show()


def test_exact_neumann_dirichlet():
    L = 10

    N = 128
    Ns = 200 + 1 #20000 + 1
    algo = 1
    scheme = 1

    W, u0, x = init_chebyshev_fredrikson(N, L)
    u0[0] = 0.

    print 'ETDRK4 N = ', N, ' Ns = ', Ns-1
    #q3, x3 = cheb_mde_neumann_dirichlet_etdrk4(W, u0, L, Ns, algo, scheme)
    lbc = BC('Neumann')
    rbc = BC('Dirichlet')
    etdrk4_solver = ETDRK4(L, N, Ns, h=None, lbc=lbc, rbc=rbc)
    q3, x3 = etdrk4_solver.solve(W, u0)
    Q3 = 0.5 * L * cheb_quadrature_clencurt(q3)
    if scheme == 0:
        #data_name = 'benchmark/NBC-DBC/exact/ETDRK4_Cox_N' 
        data_name = 'ETDRK4_Cox_N' 
        data_name = data_name + str(N) + '_Ns' + str(Ns-1)
    else:
        #data_name = 'benchmark/NBC-DBC/exact/ETDRK4_Krogstad_N' 
        data_name = 'ETDRK4_Krogstad_N' 
        data_name = data_name + str(N) + '_Ns' + str(Ns-1)
    savemat(data_name,{
            'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
            'x':x, 'q':q3, 'Q':Q3})
    plt.plot(x3, q3, 'r')
    plt.axis([0, 10, 0, 1.15])
    plt.xlabel('$z$')
    plt.ylabel('$q(z)$')
    plt.savefig(data_name, bbox_inches='tight')
    plt.show()


def test_exact_robin_dirichlet():
    L = 10.0

    N = 128
    Ns = 200 + 1 # 20000 + 1
    ka = 1.0
    algo = 1
    scheme = 1

    W, u0, x = init_chebyshev_fredrikson(N, L)
    u0[0] = 0.

    print 'ETDRK4 N = ', N, ' Ns = ', Ns-1
    #q3, x3 = cheb_mde_robin_dirichlet_etdrk4(W, u0, L, Ns, ka, algo, scheme)
    lbc = BC('Robin', (1.0, ka, 0.0))
    rbc = BC('Dirichlet')
    etdrk4_solver = ETDRK4(L, N, Ns, h=None, lbc=lbc, rbc=rbc)
    q3, x3 = etdrk4_solver.solve(W, u0)
    Q3 = 0.5 * L * cheb_quadrature_clencurt(q3)
    if scheme == 0:
        #data_name = 'benchmark/RBC-DBC/exact/ETDRK4_Cox_N' 
        data_name = 'ETDRK4_Cox_N' 
        data_name = data_name + str(N) + '_Ns' + str(Ns-1)
    else:
        #data_name = 'benchmark/RBC-DBC/exact/ETDRK4_Krogstad_N' 
        data_name = 'ETDRK4_Krogstad_N' 
        data_name = data_name + str(N) + '_Ns' + str(Ns-1)
    savemat(data_name,{
            'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
            'x':x, 'q':q3, 'Q':Q3})
    plt.plot(x3, q3, 'r')
    plt.axis([0, 10, 0, 1.15])
    plt.xlabel('$z$')
    plt.ylabel('$q(z)$')
    plt.savefig(data_name, bbox_inches='tight')
    plt.show()


def test_exact_robin():
    L = 10

    N = 128
    Ns = 200 + 1 #20000 + 1
    ka = -1. * L
    kb = 0.5 * L
    algo = 1
    scheme = 1

    W, u0, x = init_chebyshev_fredrikson(N, L)

    print 'ETDRK4 N = ', N, ' Ns = ', Ns-1
    #q3, x3 = cheb_mde_robin_etdrk4(W, u0, L, Ns, ka, kb, algo, scheme)
    lbc = BC('Robin', (1.0, ka, 0.0))
    rbc = BC('Robin', (1.0, kb, 0.0))
    etdrk4_solver = ETDRK4(L, N, Ns, h=None, lbc=lbc, rbc=rbc)
    q3, x3 = etdrk4_solver.solve(W, u0)
    Q3 = 0.5 * L * cheb_quadrature_clencurt(q3)
    if scheme == 0:
        #data_name = 'benchmark/RBC-RBC/exact/ETDRK4_Cox_N' 
        data_name = 'ETDRK4_Cox_N' 
        data_name = data_name + str(N) + '_Ns' + str(Ns-1)
    else:
        #data_name = 'benchmark/RBC-RBC/exact/ETDRK4_Krogstad_N' 
        data_name = 'ETDRK4_Krogstad_N' 
        data_name = data_name + str(N) + '_Ns' + str(Ns-1)
    savemat(data_name,{
            'N':N, 'Ns':Ns-1, 'W':W, 'u0':u0, 'Lz':L,
            'x':x, 'q':q3, 'Q':Q3})
    plt.plot(x3, q3, 'r')
    plt.axis([0, 10, 0, 1.15])
    plt.xlabel('$z$')
    plt.ylabel('$q(z)$')
    plt.savefig(data_name, bbox_inches='tight')
    plt.show()


def test_speed_space_oss():
    '''
    Confirm the complexity O(NlnN) of OSS.
    '''
    # Construct reference solution
    oscheb_ref = '../benchmark/exact/OSCHEB_N'
    oscheb_ref = oscheb_ref + '8192_Ns200000.mat'
    mat = loadmat(oscheb_ref)
    q_ref = mat['q']
    Q_ref = mat['Q'][0,0]
    N_ref = mat['N']
    Ns_ref = mat['Ns']

    L = 10
    n = 18 # Nmax = 2^n
    Ns = 200+1 # highest accuracy for reference. h = 1e-4
    M_array = np.ones(n-1) # number of same run
    M_array[:11] = 5000
    M_array[11:14] = 1000 #8192, 16384, 32768
    M_array[14] = 500 # 65536
    M_array[15] = 200 # 131072
    M_array[16] = 100 # 262144
    is_save = 1

    N_array = []
    t_full_array = [] # include initialization
    t_array = [] # do not include initialization
    err_array = []
    i = 0
    for N in 2**np.arange(2, n+1):
        M = int(M_array[i])
        W, u0, x = init_fourier(N, L)
        u0[0] = 0.; u0[N] = 0.;

        with Timer() as t:
            for m in xrange(M):
                solver = OSS(L, N, Ns)
                q, x = solver.solve(W, u0)
        t_full = t.secs / M
        t_full_array.append(t_full)

        solver = OSS(L, N, Ns)
        with Timer() as t:
            for m in xrange(M):
                q, x = solver.solve(W, u0)
        t = t.secs / M
        t_array.append(t)

        N_array.append(N)
        q.shape = (q.size,)
        Q = L * oss_integral_weights(q)
        err = np.abs(Q - Q_ref) / np.abs(Q_ref)
        err_array.append(err)
        print N, '\t', t_full_array[-1], '\t', 
        print t_array[-1], '\t', err_array[-1]
        i += 1

    if is_save:
        savemat('speed_OSS_N',{
            'N':N_array, 'Ns':Ns-1, 'N_ref':N_ref, 'Ns_ref':Ns_ref, 
            't_full':t_full_array, 't':t_array, 'err':err_array})

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(N_array, t_full_array, '.-', label='Full')
    ax.plot(N_array, t_array, '.-', label='Core')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$N$')
    plt.ylabel('Computer time')
    plt.grid('on')
    ax.legend(loc='upper left')
    if is_save:
        plt.savefig('speed_OSS_N', bbox_inches='tight')
    plt.show()

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(err_array, t_array, 'o-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative error in $Q$')
    plt.ylabel('Computer time')
    plt.grid('on')
    if is_save:
        plt.savefig('speed_error_OSS_N', bbox_inches='tight')
    plt.show()


def test_speed_accuracy_oss():
    '''
    Computation time vs. error.
    '''
    # Construct reference solution
    oscheb_ref = '../benchmark/exact/OSCHEB_N'
    oscheb_ref = oscheb_ref + '8192_Ns200000.mat'
    mat = loadmat(oscheb_ref)
    q_ref = mat['q']
    Q_ref = mat['Q'][0,0]
    N_ref = mat['N']
    Ns_ref = mat['Ns']

    L = 10
    n = 17 # Nmax = 2^n
    Ns = 20000+1 # highest accuracy for reference. h = 1e-4
    M_array = np.ones(n-1) # number of same run
    M_array[:7] = 600
    M_array[7:10] = 300 # 512, 1024, 2048
    M_array[10] = 160 # 4096
    M_array[11] = 80 # 8192
    M_array[12] = 40 #16384, 32768
    M_array[13] = 20 #16384, 32768
    M_array[14] = 10 # 65536
    M_array[15] = 3 # 131072
    is_save = 1

    N_array = []
    t_array = [] # do not include initialization
    err_array = []
    i = 0
    for N in 2**np.arange(2, n+1):
        M = int(M_array[i])
        W, u0, x = init_fourier(N, L)
        u0[0] = 0.; u0[N] = 0.;

        solver = OSS(L, N, Ns)
        with Timer() as t:
            for m in xrange(M):
                q, x = solver.solve(W, u0)
        t = t.secs / M
        t_array.append(t)

        N_array.append(N)
        q.shape = (q.size,)
        Q = L * oss_integral_weights(q)
        err = np.abs(Q - Q_ref) / np.abs(Q_ref)
        err_array.append(err)
        print N, '\t', t_array[-1], '\t', err_array[-1]
        i += 1

    if is_save:
        savemat('speed_OSS_accuracy',{
            'N':N_array, 'Ns':Ns-1, 'N_ref':N_ref, 'Ns_ref':Ns_ref, 
            't':t_array, 'err':err_array})

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(N_array, t_array, 'o-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$N$')
    plt.ylabel('Computer time')
    plt.grid('on')
    if is_save:
        plt.savefig('speed_OSS_accuracy', bbox_inches='tight')
    plt.show()

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(t_array, err_array, 'o-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Computer time')
    plt.ylabel('Relative error in $Q$')
    plt.grid('on')
    if is_save:
        plt.savefig('speed_error_OSS_accuracy', bbox_inches='tight')
    plt.show()


def test_speed_space_oscheb():
    '''
    Confirm the complexity O(NlnN) of OSCHEB.
    '''
    # Construct reference solution
    oscheb_ref = '../benchmark/exact/OSCHEB_N'
    oscheb_ref = oscheb_ref + '8192_Ns200000.mat'
    mat = loadmat(oscheb_ref)
    q_ref = mat['q']
    Q_ref = mat['Q'][0,0]
    N_ref = mat['N']
    Ns_ref = mat['Ns']

    L = 10
    n = 10 # Nmax = 2^n
    Ns = 200+1 # highest accuracy for reference. h = 1e-4
    M_array = np.ones(n-1) # number of same run
    M_array[:5] = 1000 # 4, 8, 16, 32, 64
    M_array[5] = 500 # 128
    M_array[6] = 200 # 256
    M_array[7] = 100 # 512
    M_array[8] = 50 # 1024
    is_save = 1

    N_array = []
    t_full_array = [] # include initialization
    t_array = [] # do not include initialization
    err_array = []
    i = 0
    for N in 2**np.arange(2, n+1):
        M = int(M_array[i])
        W, u0, x = init_chebyshev_fredrikson(N, L)
        u0[0] = 0.; u0[N] = 0.;

        solver = OSCHEB(L, N, Ns)
        t = clock()
        for m in xrange(M):
            q, x = solver.solve(W, u0)
        t = (clock() - t) / M
        t_array.append(t)

        t_full = clock()
        for m in xrange(M):
            solver = OSCHEB(L, N, Ns)
            q, x = solver.solve(W, u0)
        t_full = (clock() - t_full) / M
        t_full_array.append(t_full)

        N_array.append(N)
        q.shape = (q.size,)
        Q = 0.5 * L * cheb_quadrature_clencurt(q)
        err = np.abs(Q - Q_ref) / np.abs(Q_ref)
        err_array.append(err)
        print N, '\t', t_full_array[-1], '\t', 
        print t_array[-1], '\t', err_array[-1]
        i += 1

    if is_save:
        savemat('speed_OSCHEB_N',{
            'N':N_array, 'Ns':Ns-1, 'N_ref':N_ref, 'Ns_ref':Ns_ref, 
            't_full':t_full_array, 't':t_array, 'err':err_array})

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(N_array, t_full_array, '.-', label='Full')
    ax.plot(N_array, t_array, '.-', label='Core')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$N$')
    plt.ylabel('Computer time')
    plt.grid('on')
    ax.legend(loc='upper left')
    if is_save:
        plt.savefig('speed_OSCHEB_N', bbox_inches='tight')
    plt.show()

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(err_array, t_array, 'o-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Relative error in $Q$')
    plt.ylabel('Computer time')
    plt.grid('on')
    if is_save:
        plt.savefig('speed_error_OSCHEB_N', bbox_inches='tight')
    plt.show()


def test_speed_accuracy_oscheb():
    '''
    Computation time vs. error.
    '''
    # Construct reference solution
    oscheb_ref = '../benchmark/exact/OSCHEB_N'
    oscheb_ref = oscheb_ref + '8192_Ns200000.mat'
    mat = loadmat(oscheb_ref)
    q_ref = mat['q']
    Q_ref = mat['Q'][0,0]
    N_ref = mat['N']
    Ns_ref = mat['Ns']

    L = 10
    n = 10 # Nmax = 2^n
    Ns = 20000+1 
    M_array = np.ones(n-1) # number of same run
    #M_array[:5] = 1000 # 4, 8, 16, 32, 64
    #M_array[5] = 500 # 128
    #M_array[6] = 200 # 256
    #M_array[7] = 100 # 512
    #M_array[8] = 50 # 1024
    is_save = 1

    N_array = []
    t_array = [] # do not include initialization
    err_array = []
    i = 0
    for N in 2**np.arange(2, n+1):
        M = int(M_array[i])
        W, u0, x = init_chebyshev_fredrikson(N, L)
        u0[0] = 0.; u0[N] = 0.;

        solver = OSCHEB(L, N, Ns)
        t = clock()
        for m in xrange(M):
            q, x = solver.solve(W, u0)
        t = (clock() - t) / M
        t_array.append(t)

        N_array.append(N)
        q.shape = (q.size,)
        Q = 0.5 * L * cheb_quadrature_clencurt(q)
        err = np.abs(Q - Q_ref) / np.abs(Q_ref)
        err_array.append(err)
        print N, '\t', t_array[-1], '\t', err_array[-1]
        i += 1

    if is_save:
        savemat('speed_OSCHEB_accuracy',{
            'N':N_array, 'Ns':Ns-1, 'N_ref':N_ref, 'Ns_ref':Ns_ref, 
            't':t_array, 'err':err_array})

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(N_array, t_array, 'o-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$N$')
    plt.ylabel('Computer time')
    plt.grid('on')
    if is_save:
        plt.savefig('speed_OSCHEB_accuracy', bbox_inches='tight')
    plt.show()

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(t_array, err_array, 'o-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Computer time')
    plt.ylabel('Relative error in $Q$')
    plt.grid('on')
    if is_save:
        plt.savefig('speed_error_OSCHEB_accuracy', bbox_inches='tight')
    plt.show()


def test_speed_space_etdrk4():
    '''
    The expect complexity for ETDRK4 is O(N^2).
    However, due to the calculation of matrix exponential,
    it exceeds O(N^2) for large N.
    '''
    # Construct reference solution
    oscheb_ref = '../benchmark/exact/OSCHEB_N'
    oscheb_ref = oscheb_ref + '8192_Ns200000.mat'
    mat = loadmat(oscheb_ref)
    q_ref = mat['q']
    Q_ref = mat['Q'][0,0]
    N_ref = mat['N']
    Ns_ref = mat['Ns']

    L = 10.0
    n = 10 # Nmax = 2^n
    Ns = 200+1 # highest accuracy for reference. h = 1e-4
    M_array = np.ones(n-1) # number of same run
    M_array[0:5] = 1000 # 4, 8, 16, 32, 64
    M_array[5] = 500 # 128
    M_array[6] = 100 # 256
    M_array[7] = 20 # 512
    M_array[8] = 5 # 1024

    N_array = []
    t_full_array = []
    t_array = []
    err_array = []
    i = 0
    for N in 2**np.arange(2, n+1):
        M = int(M_array[i])
        W, u0, x = init_chebyshev_fredrikson(N, L)
        
        solver = ETDRK4(L, N, Ns)
        t = clock()
        for m in xrange(M):
            q, x = solver.solve(W, u0)
        t = (clock() - t) / M
        t_array.append(t)

        t_full = clock()
        for m in xrange(M):
            solver = ETDRK4(L, N, Ns)
            q, x = solver.solve(W, u0)
        t_full = (clock() - t_full) / M
        t_full_array.append(t_full)

        N_array.append(N)
        q.shape = (q.size,)
        Q = 0.5 * L * cheb_quadrature_clencurt(q)
        err = np.abs(Q - Q_ref) / np.abs(Q_ref)
        err_array.append(err)
        print N, '\t', t_full_array[-1], '\t', 
        print t_array[-1], '\t', err_array[-1]
        i += 1

    is_save = 1
    is_display = 1
    if is_save:
        savemat('speed_ETDRK4_N',{
            'N':N_array, 'Ns':Ns-1, 'N_ref':N_ref, 'Ns_ref':Ns_ref,
            't_full':t_full_array, 't':t_array, 'err':err_array})
    if is_display:
        plt.figure()
        ax = plt.subplot(111)
        ax.plot(N_array, t_full_array, '.-', label='Full')
        ax.plot(N_array, t_array, '.-', label='Core')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$N$')
        plt.ylabel('Computer time')
        plt.grid('on')
        ax.legend(loc='upper left')
        if is_save:
            plt.savefig('speed_ETDRK4_N', bbox_inches='tight')
        plt.show()

        plt.figure()
        ax = plt.subplot(111)
        ax.plot(err_array, t_array, 'o-')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Relative error in $Q$')
        plt.ylabel('Computer time')
        plt.grid('on')
        if is_save:
            plt.savefig('speed_error_ETDRK4_N', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    #test_exact_dirichlet(1,1,1)
    #test_exact_neumann(1,1,1)
    #test_exact_neumann_dirichlet()
    #test_exact_robin_dirichlet()
    #test_exact_robin()

    #test_speed_space_oss()
    #test_speed_accuracy_oss()
    #test_speed_space_oscheb()
    test_speed_accuracy_oscheb()
    #test_speed_space_etdrk4()

