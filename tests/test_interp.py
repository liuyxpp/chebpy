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

from chebpy import barycentric_weights, barycentric_weights_cg
from chebpy import barycentric_weights_cgl, barycentric_matrix
from chebpy import interpolation_point, cheb_interpolation_point
from chebpy import interpolation_1d, interpolation_2d
from chebpy import cheb_interpolation_1d, cheb_interpolation_2d
from chebpy import almost_equal

def test_barycentric_weights():
    N = 16
    ii = np.arange(N+1)
    print 'Chebyshev Gauss-Lobatto points'
    xx = np.cos(ii * np.pi / N)
    w = barycentric_weights(xx)
    w1 = barycentric_weights_cgl(N)
    print w / np.max(w)
    print w1
    print w/np.max(w) - w1
    print almost_equal(w/np.max(w),w1)

    print 'Chebyshev Gauss points'
    xx = np.cos((2*ii+1)*np.pi/(2*N+2))
    w = barycentric_weights(xx)
    w1 = barycentric_weights_cg(N)
    print w / np.max(w)
    print w1
    print w/np.max(w) - w1
    print almost_equal(w/np.max(w),w1)


def test_interpolation_1d():
    '''
    Test function
        f(x) = |x| + x/2 - x^2
    '''
    
    N = 64
    M = 1000

    ii = np.arange(N+1)
    xx = np.cos(ii * np.pi / N)
    w = barycentric_weights_cgl(N)
    f = np.abs(xx) + .5 * xx - xx**2
    plt.plot(xx,f,'.')

    kk = np.arange(M+1)
    yy = (2. / M) * kk - 1.
    f1 = np.zeros_like(yy)
    f2 = np.zeros_like(yy)
    f3 = np.zeros_like(yy)
    f4 = np.zeros_like(yy)

    print 'Interpolation by point-by-point'
    t = time()
    for j in xrange(M+1):
        f1[j] = cheb_interpolation_point(yy[j],f)
        f2[j] = interpolation_point(yy[j], f, xx, w)
    print 'point-by-point time: ', time() - t
    print f2-f1
    print almost_equal(f1,f2)
    plt.plot(yy, f1)

    print 'Interpolation by matrix'
    t = time()
    f3 = cheb_interpolation_1d(yy, f)
    f4 = interpolation_1d(yy, f, xx, w)
    print 'matrix time: ', time() - t
    print f3-f1
    print f4-f1
    print almost_equal(f3,f1), almost_equal(f4,f1)
    plt.plot(yy, f3, 'r')

    plt.show()


def test_interpolation_2d():
    '''
    Test function
        f(x,y) = x^2 + y^2
        f(x,y) = exp(-(x^2+y^2))
    '''
    
    Nx = 32
    Ny = 32
    Mx = 100
    My = 200

    ix = np.arange(Nx+1)
    x = np.cos(ix * np.pi / Nx)
    wx = barycentric_weights_cgl(Nx)
    iy = np.arange(Ny+1)
    y = np.cos(iy * np.pi / Ny)
    wy = barycentric_weights_cgl(Ny)
    X, Y = np.meshgrid(x, y)
    #f = X**2 + Y**2
    f = np.exp(-(X**2 + Y**2))
    plt.plot(X[Ny/2,:], f[Ny/2.,:],'.')

    kx = np.arange(Mx+1)
    u = (2. / Mx) * kx - 1.
    ky = np.arange(My+1)
    v = (2. / My) * ky - 1.
    U, V = np.meshgrid(u, v)
    #f0 = U**2 + V**2
    f0 = np.exp(-(U**2 + V**2))

    f1 = np.zeros_like(f)
    f2 = np.zeros_like(f)

    print 'cheb interpolation'
    t = time()
    f1 = cheb_interpolation_2d(u, v, f)
    print 'cheb time: ', time() - t

    print 'general interpolation'
    t = time()
    f2 = interpolation_2d(u, v, f, x, y, wx, wy)
    print 'general time: ', time() - t

    print f2-f1
    print 'Is general and cheb interpolation the same? ',
    print almost_equal(f1,f2)
    print 'Interpolation error: ', np.linalg.norm(f1-f0)
    plt.plot(U[My/2,:], f1[My/2.,:])

    plt.show()


if __name__ == '__main__':
    #test_barycentric_weights()
    #test_interpolation_1d()
    test_interpolation_2d()

