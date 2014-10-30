# -*- coding: utf-8 -*-
#/usr/bin/env python

import numpy as np
import matplotlib.pylab as plt

from timer import Timer

from chebpy import ETDRK4FxCy, ETDRK4FxCy2, BC, ETDRK4
from chebpy import ROBIN, DIRICHLET

def test_etdrk4():
    '''
    The test case is according to R. C. Daileda Lecture notes.
            du/dt = (1/25) u_xx , x@(0,3)
    with boundary conditions:
            u(0,t) = 0
            u_x(3,t) = -(1/2) u(3,t)
            u(x,0) = 100*(1-x/3)
    Conclusion:
        We find that the numerical solution is much more accurate than the five
        term approximation of the exact analytical solution.
    '''
    Nx = 64
    Lx = 3
    t = 1.
    Ns = 101
    ds = t/(Ns - 1)

    ii = np.arange(Nx+1)
    x = np.cos(np.pi * ii / Nx) # yy [-1, 1]
    x = 0.5 * (x + 1) * Lx # mapping to [0, Ly]
    w = np.zeros(Nx+1)
    q = np.zeros([Ns, Nx+1])
    q[0] = 100*(1-x/3)
    # The approximation of exact solution by first 5 terms
    q_exact = 47.0449*np.exp(-0.0210*t)*np.sin(0.7249*x) + \
              45.1413*np.exp(-0.1113*t)*np.sin(1.6679*x) + \
              21.3586*np.exp(-0.2872*t)*np.sin(2.6795*x) + \
              19.3403*np.exp(-0.5505*t)*np.sin(3.7098*x) + \
              12.9674*np.exp(-0.9015*t)*np.sin(4.7474*x)
    lbc = BC(DIRICHLET, [0,1,0])
    rbc = BC(ROBIN, [1.,0.5,0])

    q_solver = ETDRK4(Lx,Nx,Ns,h=ds,c=1./25,lbc=lbc,rbc=rbc)
    q1, x = q_solver.solve(w, q[0], q)
    plt.plot(x, q[0], label='q_0')
    plt.plot(x, q1, label='q_solution')
    plt.plot(x, q_exact, label='q_exact')
    plt.legend(loc='best')
    plt.show()


def test_etdrk4_complex():
    '''
    The test case is according to R. C. Daileda Lecture notes.
            du/dt = (1/25) u_xx , x@(0,3)
    with boundary conditions:
            u(0,t) = 0
            u_x(3,t) = -(1/2) u(3,t)
            u(x,0) = 100*(1-x/3)
    Conclusion:
        We find that the numerical solution is much more accurate than the five
        term approximation of the exact analytical solution.
    '''
    Nx = 64
    Lx = 3
    t = 1.
    Ns = 101
    ds = t/(Ns - 1)

    ii = np.arange(Nx+1)
    x = np.cos(np.pi * ii / Nx) # yy [-1, 1]
    x = 0.5 * (x + 1) * Lx # mapping to [0, Ly]
    w = np.zeros(Nx+1, dtype=np.complex128)
    q = np.zeros([Ns, Nx+1], dtype=np.complex128)
    q[0] = 100*(1-x/3)
    # The approximation of exact solution by first 5 terms
    q_exact = 47.0449*np.exp(-0.0210*t)*np.sin(0.7249*x) + \
              45.1413*np.exp(-0.1113*t)*np.sin(1.6679*x) + \
              21.3586*np.exp(-0.2872*t)*np.sin(2.6795*x) + \
              19.3403*np.exp(-0.5505*t)*np.sin(3.7098*x) + \
              12.9674*np.exp(-0.9015*t)*np.sin(4.7474*x)
    lbc = BC(DIRICHLET, [0,1,0])
    rbc = BC(ROBIN, [1.,0.5,0])

    q_solver = ETDRK4(Lx,Nx,Ns,h=ds,c=1./25,lbc=lbc,rbc=rbc)
    q1, x = q_solver.solve(w, q[0], q)
    plt.plot(x, q[0].real, label='q_0')
    plt.plot(x, q1.real, label='q_solution1')
    plt.plot(x, q[-1].real, label='q_solution2')
    plt.plot(x, q_exact, label='q_exact')
    plt.legend(loc='best')
    plt.show()


def check(u):
    '''
        The PDE is
            du/dt = u_xx + u_yy - wu
        Calculate the residual using FD scheme.
            R(x) = (u(x+h)
    '''
    pass

if __name__ == '__main__':
    #test_etdrk4()
    test_etdrk4_complex()
