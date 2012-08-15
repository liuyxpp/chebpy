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
from chebpy import cheb_allen_cahn_etdrk4

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
    Ns = 201

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

    q2, x2 = cheb_mde_etdrk4(W, L, Ns)

    plt.plot(x1, q1)
    plt.plot(x1, q1, '.')
    plt.plot(x2, q2, 'r')
    plt.plot(x2, q2, 'r.')
    plt.axis([0, 10, 0, 1.15])
    plt.show()


def test_cheb_allen_cahn_etdrk4():
    cheb_allen_cahn_etdrk4()


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
    test_cheb_mde()
    #test_cheb_allen_cahn_etdrk4()
    #test_speed()

