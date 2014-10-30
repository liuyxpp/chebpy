# -*- coding: utf-8 -*-
#/usr/bin/env python

import numpy as np
import matplotlib.pylab as plt

from timer import Timer

from chebpy import ETDRK4FxCy, ETDRK4FxCy2, BC, ETDRK4
from chebpy import ROBIN, DIRICHLET

def test_etdrk4fxcy():
    '''
    The test function is
            u = e^[f(x,y) - t]
    where
            f(x,y) = h(x) + g(y)
    Assume it is the solution of following PDE
            du/dt = (d^2/dx^2 + d^2/dy^2) u - w(x,y)u
    in the domain [0,Lx]x[0,Ly] for time t=0 to t=1,
    with boundary conditions
            u(x+Lx,y,t) = u(x,y,t) # periodic in x direction
            d/dy[u(x,y=0,t)] = ka u(y=0)
            d/dy[u(x,y=Ly,t)] = kb u(y=Ly)
    To generate a suitable solution, we assume
            h(x) = sin(x)
            h_x = dh/dx = cos(x)
            h_xx = d^2h/dx^2 = -sin(x)
    since it is periodic in x direction.
    The corresponding w(x,y) is
            w(x,y) = h_xx + g_yy + (h_x)^2 + (g_y)^2 + 1

    1. For homogeneous NBC (ka=kb=0), a suitable g(y) is
            g(y) = Ay^2(2y-3)/6
            g_y = A(y^2-y)  # g_y(y=0)=0, g_y(y=1)=0
            g_yy = A(2*y-1)
    where A is a positive constant.
        Lx = 2*pi, Ly = 1.0, Nx = 64, Ny =32, Ns = 21
    is a good parameter set. Note the time step ds = 1/(Ns-1) = 0.05 is very
    large.
    2. For homogeneous DBC, an approximate g(y) is
            g(y) = -A(y-1)^2
            g_y = -2A(y-1)
            g_yy = -2A
    where A is a positive and large constant.
        Lx = 2*pi, Ly = 2.0, Nx = 64, Ny =32, Ns = 101
    is a good parameter set.
    3. For RBC, g(y) is given by
            g(y) = -Ay
            g_y = -A # ka=kb=-A
            g_yy = 0
       A is a positive constant.
       Numerical result is different than the analytical one.
    '''
    Lx = 2*np.pi # x [0, Lx]
    Nx = 64
    Ly = 1.0 # y [0, Ly]
    Ny = 127
    Ns = 101
    ds = 1. / (Ns - 1)

    # Periodic in x direction, Fourier
    xx = np.arange(Nx) * Lx / Nx
    # Non-periodic in y direction, Chebyshev
    ii = np.arange(Ny+1)
    yy = np.cos(np.pi * ii / Ny) # yy [-1, 1]
    yy = 0.5 * (yy + 1) * Ly # mapping to [0, Ly]
    w = np.zeros([Nx,Ny+1])
    A = 1.0
    q = np.zeros([Ns, Nx, Ny+1])
    q_exact = np.zeros([Nx, Ny+1])
    #q[0] = 1.
    for i in xrange(Nx):
        for j in xrange(Ny+1):
            x = xx[i]
            y = yy[j]
            # RBC
            #q_exact[i,j] = np.exp(-A*y + np.sin(x) - 1)
            #q[0,i,j] = np.exp(-A*y + np.sin(x))
            #w[i,j] = np.cos(x)**2 - np.sin(x) + A**2 + 1
            # homogeneous NBC
            q_exact[i,j] = np.exp(A*y**2*(2*y-3)/6 + np.sin(x) - 1)
            q[0,i,j] = np.exp(A*y**2*(2*y-3)/6 + np.sin(x))
            w[i,j] = (A*y*(y-1))**2 + np.cos(x)**2 - np.sin(x) + A*(2*y-1) + 1
            # homogeneous DBC
            #q[0,i,j] = np.exp(-A*(y-1)**2 + np.sin(x))
            #q_exact[i,j] = np.exp(-A*(y-1)**2 + np.sin(x) + 1)
            #w[i, j] = np.cos(x)**2 - np.sin(x) + 4*A**2 + (2*A*(y-1))**2 + 1
            # Fredrickson
            #sech = 1. / np.cosh(0.25*(6*y[j]-3*Ly))
            #w[i,j] = (1 - 2*sech**2)*(np.sin(2*np.pi*x[i]/Lx)+1)
            #w[i,j] = (1 - 2*sech**2)

    x = xx; y = yy
    plt.imshow(w)
    plt.xlabel('w')
    plt.show()
    plt.plot(x,w[:,Ny/2])
    plt.xlabel('w(x)')
    plt.show()
    plt.plot(y,w[Nx/4,:])
    plt.xlabel('w(y)')
    plt.show()

    # DBC
    #lbc = BC(DIRICHLET, [0.0, 1.0, 0.0])
    #rbc = BC(DIRICHLET, [0.0, 1.0, 0.0])
    # RBC
    #lbc = BC(ROBIN, [1.0, A, 0.0])
    #rbc = BC(ROBIN, [1.0, A, 0.0])
    # NBC
    lbc = BC(ROBIN, [1.0, 0, 0.0])
    rbc = BC(ROBIN, [1.0, 0, 0.0])
    #q_solver = ETDRK4FxCy(Lx, Ly, Nx, Ny, Ns, h=ds, lbc=lbc, rbc=rbc)
    q_solver = ETDRK4FxCy2(Lx, Ly, Nx, Ny, Ns, h=ds, lbc=lbc, rbc=rbc)
    M = 100   # Took 1117.6 x 4 seconds for cpu one core
    with Timer() as t:
        for m in xrange(M):
            q1 = q_solver.solve(w, q[0], q)
    print "100 runs took ", t.secs, " seconds."

    print 'Error =', np.max(np.abs(q1-q_exact))

    plt.imshow(q[0])
    plt.xlabel('q_0')
    plt.show()
    plt.imshow(q1)
    plt.xlabel('q_solution')
    plt.show()
    plt.imshow(q_exact)
    plt.xlabel('q_exact')
    plt.show()
    plt.plot(x,q[0,:,Ny/2], label='q0')
    plt.plot(x,q1[:,Ny/2], label='q_solution')
    plt.plot(x,q_exact[:,Ny/2], label='q_exact')
    plt.legend(loc='best')
    plt.xlabel('q[:,Ny/2]')
    plt.show()
    plt.plot(y,q[0,Nx/4,:], label='q0')
    plt.plot(y,q1[Nx/4,:], label='q_solution')
    plt.plot(y,q_exact[Nx/4,:], label='q_exact')
    plt.legend(loc='best')
    plt.xlabel('q[Nx/4,:]')
    plt.show()
    plt.plot(y,q[0,Nx*3/4,:], label='q0')
    plt.plot(y,q1[Nx*3/4,:], label='q_solution')
    plt.plot(y,q_exact[Nx*3/4,:], label='q_exact')
    plt.legend(loc='best')
    plt.xlabel('q[Nx*3/4,:]')
    plt.show()
    exit()

    # Check with ETDRK4
    sech = 1. / np.cosh(0.25*(6*y-3*Ly))
    w1 = 1 - 2*sech**2
    plt.plot(y,w1)
    plt.show()
    q = np.zeros([Ns, Ny+1])
    q[0] = 1.
    q_solver = ETDRK4(Ly,Ny,Ns,h=ds,lbc=lbc,rbc=rbc)
    q1, y = q_solver.solve(w1, q[0], q)
    plt.plot(y,q1)
    plt.show()

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


def check(u):
    '''
        The PDE is
            du/dt = u_xx + u_yy - wu
        Calculate the residual using FD scheme.
            R(x) = (u(x+h)
    '''
    pass

if __name__ == '__main__':
    test_etdrk4fxcy()
    #test_etdrk4()

