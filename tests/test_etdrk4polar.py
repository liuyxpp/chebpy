# -*- coding: utf-8 -*-
#/usr/bin/env python

import numpy as np
import matplotlib.pylab as plt

from chebpy import ETDRK4Polar, BC
from chebpy import ROBIN, DIRICHLET

from scftpy import scft_contourf

def test_etdrk4polar():
    '''
    The PDE is
            du/dt = (d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/dtheta^2) u - w u
    in the domain [0,R]x[0,2*pi] for time t=0 to t=1,
    with boundary conditions
            u(r,theta,t) = u(r,theta+2*pi,t) # periodic in theta direction
            du/dr |{r=1} + ka u(r=1) = 0, # RBC
    '''
    Nt = 64 # theta
    R = 1.0 # r [0, R]
    Nr = 63 # Nr must be odd
    N2 = (Nr+1) / 2
    Ns = 101 
    ds = 1. / (Ns - 1)
    
    # Periodic in x direction, Fourier
    tt = np.arange(Nt+1) * 2 * np.pi / Nt
    # Non-periodic in y direction, Chebyshev
    ii = np.arange(Nr+1)
    rr = np.cos(np.pi * ii / Nr) # rr [-1, 1]
    rr = rr[:N2] # rr in (0, 1] with r[0] = 1
    w = np.zeros([Nt,N2])
    A = 1.0
    q = np.zeros([Ns, Nt, N2])
    r, t = np.meshgrid(rr, tt[:-1])
    rp, tp = np.meshgrid(rr, tt)
    q[0] = np.exp(-A*r**2) * np.sin(t)
    w = np.cos(t)
    #plt.contourf(r*np.cos(t), r*np.sin(t), q[0])
    q0p = np.zeros([Nt+1,N2])
    q0p[:-1,:] = q[0]
    q0p[-1,:] = q[0,0,:]
    print rp.shape, tp.shape, q0p.shape
    scft_contourf(rp*np.cos(tp), rp*np.sin(tp), q0p)
    plt.xlabel('q[0]')
    plt.show()
    #plt.contourf(r*np.cos(t), r*np.sin(t), w)
    wp = np.zeros([Nt+1,N2])
    wp[:-1,:] = w
    wp[-1,:] = w[0,:]
    scft_contourf(rp*np.cos(tp), rp*np.sin(tp), wp)
    plt.xlabel('w')
    plt.show()

    #lbc = BC(DIRICHLET, [0.0, 1.0, 0.0]) 
    #rbc = BC(DIRICHLET, [0.0, 1.0, 0.0]) 
    lbc = BC(ROBIN, [1.0, -0.1, 0.0]) 
    rbc = BC(ROBIN, [1.0, 0.1, 0.0]) 
    q_solver = ETDRK4Polar(R, Nr, Nt, Ns, h=ds, lbc=lbc, rbc=rbc)
    q1 = q_solver.solve(w, q[0], q)

    #plt.contourf(r*np.cos(t), r*np.sin(t), q1)
    q1p = np.zeros([Nt+1,N2])
    q1p[:-1,:] = q1
    q1p[-1,:] = q1[0,:]
    scft_contourf(rp*np.cos(tp), rp*np.sin(tp), q1p)
    plt.xlabel('q_solution')
    plt.show()
    plt.plot(rr, q1[0,:], label='$\\theta=0$')
    plt.plot(rr, q1[Nt/8,:], label='$\\theta=\pi/4$')
    plt.plot(rr, q1[Nt/4,:], label='$\\theta=\pi/2$')
    plt.plot(rr, q1[3*Nt/8,:], label='$\\theta=3\pi/4$')
    plt.plot(rr, q1[Nt/2,:], label='$\\theta=\pi$')
    plt.plot(rr, q1[5*Nt/8,:], label='$\\theta=5\pi/4$')
    plt.plot(rr, q1[3*Nt/4,:], label='$\\theta=3\pi/2$')
    plt.plot(rr, q1[7*Nt/8,:], label='$\\theta=7\pi/4$')
    plt.legend(loc='best')
    plt.show()

def my_contourf(x, y, z, levels=None, cmap=None, **kwargs):
    dx = x.max() - x.min()
    dy = y.max() - y.min()
    w, h = plt.figaspect(float(dy/dx)) # float is must
    # No frame, white background, w/h aspect ratio figure
    fig = plt.figure(figsize=(w/2,h/2), frameon=False, dpi=150, facecolor='w')
    # full figure subplot, no boarder, no axes
    ax = fig.add_axes([0,0,1,1], frameon=False, axisbg='w')
    # no ticks
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Default: there are 256 contour levels
    if levels is None:
        step = (z.max() - z.min()) / 32
        levels = np.arange(z.min(), z.max()+step, step)
    # Default: colormap is Spectral from matplotlib.cm
    if cmap is None:
        cmap = plt.cm.Greens
    # actual plot
    ax.contourf(x, y, z, levels=levels, cmap=cmap, 
                antialiased=False, **kwargs)


if __name__ == '__main__':
    test_etdrk4polar()

