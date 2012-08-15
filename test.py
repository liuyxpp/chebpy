# -*- coding: utf-8 -*-
#/usr/bin/env python
"""
chebpy.test
===========

"""

from time import time
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

from chebpy import cheb_D1_mat, cheb_D1_fft, cheb_D1_fchebt, cheb_D1_dct
from chebpy import cheb_fast_transform, cheb_inverse_fast_transform
from chebpy import cheb_mde_splitting_pseudospectral, cheb_mde_etdrk4
from chebpy import cheb_allen_cahn_etdrk4, complex_contour_integral

def test_cheb_D1_mat():
    '''
    Example is from p18.m of Trefethen's book, p.81.
    '''

    xx = np.arange(-1, 1, .01)
    ff = np.exp(xx) * np.sin(5.*xx)
    for N in [10,20]:
        D, x = cheb_D1_mat(N)
        f = np.exp(x) * np.sin(5.*x)
        plt.figure()
        plt.plot(x, f, '.')
        plt.plot(xx, ff, 'r-')
        plt.show()
        
        err = np.dot(D, f) - np.exp(x) * (np.sin(5.*x) + 5.*np.cos(5.*x))
        plt.figure()
        plt.plot(x, err, '.')
        plt.plot(x, err, '-')
        plt.show()

def test_cheb_D1_fft():
    '''
    Example is from p18.m of Trefethen's book, p.81.
    '''

    xx = np.arange(-1, 1, .01)
    ff = np.exp(xx) * np.sin(5.*xx)
    for N in [10,20]:
        ii = np.arange(N+1)
        x = np.cos(np.pi * ii / N)
        f = np.exp(x) * np.sin(5.*x)
        plt.figure()
        plt.plot(x, f, '.')
        plt.plot(xx, ff, 'r-')
        plt.show()
        
        err = cheb_D1_fft(f) - np.exp(x) * (np.sin(5.*x) + 5.*np.cos(5.*x))
        plt.figure()
        plt.plot(x, err, '.')
        plt.plot(x, err, '-')
        plt.show()


def test_cheb_D1_fchebt():
    '''
    Example is from p18.m of Trefethen's book, p.81.
    '''

    xx = np.arange(-1, 1, .01)
    ff = np.exp(xx) * np.sin(5.*xx)
    for N in [10,20]:
        ii = np.arange(N+1)
        x = np.cos(np.pi * ii / N)
        f = np.exp(x) * np.sin(5.*x)
        plt.figure()
        plt.plot(x, f, '.')
        plt.plot(xx, ff, 'r-')
        plt.show()
        
        err = cheb_D1_fchebt(f) - np.exp(x) * (np.sin(5.*x) + 5.*np.cos(5.*x))
        plt.figure()
        plt.plot(x, err, '.')
        plt.plot(x, err, '-')
        plt.show()


def test_cheb_D1_dct():
    '''
    Example is from p18.m of Trefethen's book, p.81.
    '''

    xx = np.arange(-1, 1, .01)
    ff = np.exp(xx) * np.sin(5.*xx)
    for N in [10,20]:
        ii = np.arange(N+1)
        x = np.cos(np.pi * ii / N)
        f = np.exp(x) * np.sin(5.*x)
        plt.figure()
        plt.plot(x, f, '.')
        plt.plot(xx, ff, 'r-')
        plt.show()
        
        err = cheb_D1_dct(f) - np.exp(x) * (np.sin(5.*x) + 5.*np.cos(5.*x))
        plt.figure()
        plt.plot(x, err, '.')
        plt.plot(x, err, '-')
        plt.show()

def test_cheb_fast_transform():
    N = 10
    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    f = np.exp(x) * np.sin(5.*x)

    F = cheb_fast_transform(f)
    ff = cheb_inverse_fast_transform(F)

    print f
    print ff


def test_cheb_mde():
    L = 10
    N = 64
    Ns = 1601

    ii = np.arange(N+1)
    x = 1. * ii * L / N
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W = 1. - 2. * sech * sech
    plt.plot(x, W)
    plt.axis([0, 10, -1.1, 1.1,])
    plt.show()

    q1, x1 = cheb_mde_splitting_pseudospectral(W, L, Ns)

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


def test_accuracy_cheb_mde():
    L = 10
    N = 64
    Ns_ref = 20000+1 # highest accuracy for reference. h = 1e-4

    ii = np.arange(N+1)
    x = 1. * ii * L / N
    sech = 1. / np.cosh(.75 * (2.*x - L))
    W1 = 1. - 2. * sech * sech

    q1_ref, x1 = cheb_mde_splitting_pseudospectral(W1, L, Ns_ref)

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


def test_speed():
    N = 32
    M = 10000
    ii = np.arange(N+1)
    x = np.cos(np.pi * ii / N)
    f = np.exp(x) * np.sin(5.*x)

    t = time()
    for i in xrange(M):
        D, x = cheb_D1_mat(N)
        w = np.dot(D, f)
    print 'Run time for mat is:', time() - t

    t = time()
    for i in xrange(M):
        w = cheb_D1_fft(f)
    print 'Run time for fft is:', time() - t

    t = time()
    for i in xrange(M):
        w = cheb_D1_fchebt(f)
    print 'Run time for fchebt is:', time() - t

    t = time()
    for i in xrange(M):
        w = cheb_D1_dct(f)
    print 'Run time for dct is:', time() - t


if __name__ == '__main__':
    #test_cheb_D1_mat()
    #test_cheb_D1_fft()
    #test_cheb_D1_fchebt()
    #test_cheb_D1_dct()
    #test_cheb_fast_transform()
    #test_cheb_mde()
    #test_accuracy_cheb_mde()
    #test_cheb_allen_cahn_etdrk4()
    #test_speed()
    test_complex_contour_integral()

