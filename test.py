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
    test_speed()

