# -*- coding: utf-8 -*-
#/usr/bin/env python
"""
chebpy.test
===========

"""

from time import time
import numpy as np
from scipy.special import erf
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

from chebpy import clencurt_weights, clencurt_weights_fft
from chebpy import cheb_quadrature_clencurt, cheb_quadrature_cgl
from chebpy import almost_equal

def test_clencurt_weights():
    N = 257
    M = 500
    
    t = time()
    for i in xrange(M):
        w1 = clencurt_weights(N)
    print 'direct sum time: ', time() - t

    t = time()
    for i in xrange(M):
        w2 = clencurt_weights_fft(N)
    print 'fft time: ', time() - t

    print w1-w2
    print almost_equal(w1, w2)


def test_quadrature_clencurt():
    '''
    Four funcitons in [-1, 1] are tested:
        f(x) = |x|^3,       I = .5
        f(x) = exp(-x^(-2)),   I = 2*(exp(-1) + sqrt(pi)*(erf(1) - 1))
        f(x) = 1/(1+x^2),   I = pi/2
        f(x) = x^10,        I = 2/11
    '''

    Nmax = 25
    xN = np.arange(2,Nmax+1)
    E1 = np.zeros(Nmax-1)
    I1 = .5
    E2 = np.zeros(Nmax-1)
    I2 = 2 * (np.exp(-1) + np.sqrt(np.pi)*(erf(1) - 1))
    E3 = np.zeros(Nmax-1)
    I3 = .5 * np.pi
    E4 = np.zeros(Nmax-1)
    I4 = 2./11
    for N in xrange(2, Nmax+1):
        theta = np.arange(N+1) * np.pi / N
        x = np.cos(theta)
        f1 = np.abs(x)**3
        E1[N-2] = np.abs(cheb_quadrature_clencurt(f1) - I1)
        f2 = np.exp(-x**(-2))
        E2[N-2] = np.abs(cheb_quadrature_clencurt(f2) - I2)
        f3 = 1. / (1 + x**2)
        E3[N-2] = np.abs(cheb_quadrature_clencurt(f3) - I3)
        f4 = x**10
        E4[N-2] = np.abs(cheb_quadrature_clencurt(f4) - I4)

    plt.semilogy(xN, E1+1e-100, '.')
    plt.semilogy(xN, E1+1e-100)
    plt.axis([0, Nmax+2, 1e-18, 1e3])
    plt.grid('on')
    plt.show()

    plt.semilogy(xN, E2+1e-100, '.')
    plt.semilogy(xN, E2+1e-100)
    plt.axis([0, Nmax+2, 1e-18, 1e3])
    plt.grid('on')
    plt.show()

    plt.semilogy(xN, E3+1e-100, '.')
    plt.semilogy(xN, E3+1e-100)
    plt.axis([0, Nmax+2, 1e-18, 1e3])
    plt.grid('on')
    plt.show()

    plt.semilogy(xN, E4+1e-100, '.')
    plt.semilogy(xN, E4+1e-100)
    plt.axis([0, Nmax+2, 1e-18, 1e3])
    plt.grid('on')
    plt.show()


def test_quadrature_cgl():
    '''
    Four funcitons in [-1, 1] are tested:
        f(x) = |x|^3,       I = .5
        f(x) = exp(-x^(-2)),   I = 2*(exp(-1) + sqrt(pi)*(erf(1) - 1))
        f(x) = 1/(1+x^2),   I = pi/2
        f(x) = x^10,        I = 2/11
    '''

    Nmax = 50
    xN = np.arange(2,Nmax+1)
    E1 = np.zeros(Nmax-1)
    I1 = .5
    E2 = np.zeros(Nmax-1)
    I2 = 2 * (np.exp(-1) + np.sqrt(np.pi)*(erf(1) - 1))
    E3 = np.zeros(Nmax-1)
    I3 = .5 * np.pi
    E4 = np.zeros(Nmax-1)
    I4 = 2./11
    for N in xrange(2, Nmax+1):
        theta = np.arange(N+1) * np.pi / N
        x = np.cos(theta)
        f1 = np.abs(x)**3
        E1[N-2] = np.abs(cheb_quadrature_cgl(f1) - I1)
        print cheb_quadrature_cgl(f1), I1
        f2 = np.exp(-x**(-2))
        E2[N-2] = np.abs(cheb_quadrature_cgl(f2) - I2)
        print cheb_quadrature_cgl(f2), I2
        f3 = 1. / (1 + x**2)
        E3[N-2] = np.abs(cheb_quadrature_cgl(f3) - I3)
        print cheb_quadrature_cgl(f3), I3
        f4 = x**10
        E4[N-2] = np.abs(cheb_quadrature_cgl(f4) - I4)
        print cheb_quadrature_cgl(f4), I4

    plt.semilogy(xN, E1+1e-100, '.')
    plt.semilogy(xN, E1+1e-100)
    plt.axis([0, Nmax+2, 1e-18, 1e3])
    plt.grid('on')
    plt.show()

    plt.semilogy(xN, E2+1e-100, '.')
    plt.semilogy(xN, E2+1e-100)
    plt.axis([0, Nmax+2, 1e-18, 1e3])
    plt.grid('on')
    plt.show()

    plt.semilogy(xN, E3+1e-100, '.')
    plt.semilogy(xN, E3+1e-100)
    plt.axis([0, Nmax+2, 1e-18, 1e3])
    plt.grid('on')
    plt.show()

    plt.semilogy(xN, E4+1e-100, '.')
    plt.semilogy(xN, E4+1e-100)
    plt.axis([0, Nmax+2, 1e-18, 1e3])
    plt.grid('on')
    plt.show()


if __name__ == '__main__':
    test_quadrature_clencurt()
    #test_quadrature_cgl()
    #test_clencurt_weights()

