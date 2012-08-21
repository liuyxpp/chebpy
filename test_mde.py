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

from chebpy import cheb_mde_split, cheb_mde_neumann_split
from chebpy import cheb_mde_etdrk4, cheb_mde_neumann_etdrk4
from chebpy import cheb_mde_robin_etdrk4
from chebpy import cheb_allen_cahn_etdrk4, complex_contour_integral

def test_cheb_mde():
    L = 10
    N = 256
    Ns = 1601

    ii = np.arange(N+1)
    x = 1. * ii * L / N
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech
    plt.plot(x, W)
    plt.axis([0, 10, -1.1, 1.1])
    plt.show()

    q1, x1 = cheb_mde_split(W, L, Ns)

    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech
    plt.plot(x, W)
    plt.axis([0, 10, -1.1, 1.1,])
    plt.show()

    Ns = 11
    q2, x2 = cheb_mde_etdrk4(W, L, Ns)

    plt.plot(x1, q1)
    #plt.plot(x1, q1, '.')
    plt.plot(x2, q2, 'r')
    #plt.plot(x2, q2, 'r.')
    plt.axis([0, 10, 0, 1.15])
    plt.show()


def test_cheb_mde_neumann():
    L = 10
    N = 256
    Ns = 101

    ii = np.arange(N+1)
    x = 1. * ii * L / N
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech
    plt.plot(x, W)
    plt.axis([0, 10, -1.1, 1.1,])
    plt.show()

    q1, x1 = cheb_mde_neumann_split(W, L, Ns)

    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech
    plt.plot(x, W)
    plt.axis([0, 10, -1.1, 1.1,])
    plt.show()

    q2, x2 = cheb_mde_neumann_etdrk4(W, L, Ns)
    q3, x3 = cheb_mde_robin_etdrk4(W, L, Ns, 0., 0.)
    print np.sum(np.abs(q2-q3))/N

    plt.plot(x1, q1)
    #plt.plot(x1, q1, '.')
    plt.plot(x2, q2, 'r')
    #plt.plot(x2, q2, 'r.')
    plt.plot(x3, q3, 'g')
    plt.axis([0, 10, 0, 1.15])
    plt.show()


def test_cheb_mde_robin():
    L = 10
    N = 64
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

    q1, x1 = cheb_mde_robin_etdrk4(W, L, Ns, 0., 0.)
    q2, x2 = cheb_mde_robin_etdrk4(W, L, Ns, .5*L*ka, .5*L*kb)

    plt.plot(x1, q1, 'b')
    plt.plot(x2, q2, 'r')
    plt.axis([0, 10, 0, 1.15])
    plt.show()


def test_accuracy_cheb_mde():
    L = 10
    N = 64
    Ns_ref = 20000+1 # highest accuracy for reference. h = 1e-4

    ii = np.arange(N+1)
    x = 1. * ii * L / N
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W1 = 1. - 2. * sech * sech

    q1_ref, x1 = cheb_mde_split(W1, L, Ns_ref)

    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W2 = 1. - 2. * sech * sech

    q2_ref, x2 = cheb_mde_etdrk4(W2, L, Ns_ref)

    plt.plot(x1, q1_ref)
    plt.plot(x2, q2_ref, 'r')
    plt.show()

    # Ns = 10^t
    errs1 = []
    errs2 = []
    Nss = []
    for Ns in np.round(np.power(10, np.linspace(0,4,10))):
        Ns = int(Ns) + 1
        q1, x1 = cheb_mde_splitting_pseudospectral(W1, L, Ns)
        q2, x2 = cheb_mde_etdrk4(W2, L, Ns)
        err1 = np.max(q1 - q1_ref) / np.max(q1_ref)
        err2 = np.max(q2 - q2_ref) / np.max(q2_ref)
        Nss.append(1./Ns)
        errs1.append(err1)
        errs2.append(err2)

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

    q1_ref, x1 = cheb_mde_neumann_split(W1, L, Ns_ref)

    x = np.cos(np.pi * ii / N)
    x = .5 * (x + 1) * L
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W2 = 1. - 2. * sech * sech

    q2_ref, x2 = cheb_mde_neumann_etdrk4(W2, L, Ns_ref)
    q3_ref, x3 = cheb_mde_robin_etdrk4(W2, L, Ns_ref, 0., 0.)

    plt.plot(x1, q1_ref)
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
        q1, x1 = cheb_mde_neumann_split(W1, L, Ns)
        q2, x2 = cheb_mde_neumann_etdrk4(W2, L, Ns)
        q3, x3 = cheb_mde_robin_etdrk4(W2, L, Ns, 0., 0.)
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


def test_cheb_allen_cahn_etdrk4():
    cheb_allen_cahn_etdrk4()


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
    #test_cheb_mde()
    #test_cheb_mde_neumann()
    #test_cheb_mde_robin()
    #test_accuracy_cheb_mde()
    #test_accuracy_cheb_mde_neumann()
    test_accuracy_cheb_mde_robin()
    #test_cheb_allen_cahn_etdrk4()
    #test_complex_contour_integral()

