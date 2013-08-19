# -*- coding: utf-8 -*-
#/usr/bin/env python

import numpy as np
import matplotlib.pylab as plt

from chebpy import OSF

def test_osf():
    L = 4.0
    Nx = 128
    Ns = 101
    ds = 1. / (Ns - 1)
    
    x = np.arange(Nx) * L / Nx
    sech = 1. / np.cosh(0.25*(6*x-3*L))
    w = 1 - 2*sech**2
    plt.plot(x,w)
    plt.show()

    q = np.zeros([Ns, Nx])
    q[0, :] = 1.
    q_solver = OSF(L, Nx, Ns, ds)
    q1 = q_solver.solve(w, q[0], q)
    plt.plot(x, q[1])
    plt.plot(x, q[Ns/2])
    plt.plot(x, q[-1])
    plt.plot(x, q1)
    plt.show()

if __name__ == '__main__':
    test_osf()

