# -*- coding: utf-8 -*-
#/usr/bin/env python

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pylab as plt
from mayavi import mlab

from chebpy import ETDRK4Cylind, BC
from chebpy import ROBIN, DIRICHLET, cheb_barycentric_matrix

from scftpy import scft_contourf

def test_etdrk4cylind():
    '''
    Assume the following PDE in cylindrical coordinates,
            du/dt = (d^2/dr^2 + (1/r)d/dr + (1/r^2)d^2/d^theta + d^2/dz^2) u -
            w*u
    with u = u(r,theta,z), w=w(r,theta,z) in the domain 
            [0, R] x [0, 2pi] x [0, Lz]
    for time t=0 to t=1,
    with boundary conditions
            d/dr[u(r=R,theta,z,t)] = ka u(r=R)
            u(r,theta,z,t) = u(r,theta+2pi,z,t) # periodic in theta direction
            u(r,theta,z,t) = u(r,theta,z+lz,t) # periodic in z direction
    '''
    Lt = 2*np.pi
    Nt = 32 # theta
    Lz = 4.0
    Nz = 64
    R = 2.0 # r [0, R]
    Nr = 23 # Nr must be odd
    N2 = (Nr + 1) / 2
    Nxp = 20
    Nyp = 20
    Nzp = 15
    Ns = 101 
    ds = 1. / (Ns - 1)
    
    # Periodic in x and z direction, Fourier
    tt = np.arange(Nt) * Lt / Nt
    zz = np.arange(Nz) * Lz / Nz
    # Non-periodic in r direction, Chebyshev
    ii = np.arange(Nr+1)
    rr = np.cos(np.pi * ii / Nr) # rr [-1, 1]
    rr = rr[:N2] # rr in (0, 1] with r[0] = 1
    rrp = np.linspace(0, R, Nxp)
    A = 1.0
    q = np.zeros([Ns, Nt, Nz, N2])
    z, t, r = np.meshgrid(zz, tt, rr)
    zp, tp, rp = np.meshgrid(zz, tt, rrp)
    print tt.shape, zz.shape, rr.shape
    print t.shape, z.shape, r.shape
    print q[0].shape
    q[0] = r**2 * np.cos(t) + A*z
    xp = rp * np.cos(tp)
    yp = rp * np.sin(tp)
    print xp.max(), xp.min()
    print yp.max(), yp.min()
    print zp.max(), zp.min()
    q0 = cheb_interp3d_r(q[0], rrp)
    xpp, ypp, zpp = np.mgrid[-R:R:Nxp*1j,
                    -R:R:Nyp*1j,
                    0:Lz*(1-1./Nz):Nzp*1j]
    q0p = griddata((xp.ravel(),yp.ravel(),zp.ravel()), q0.ravel(), 
                   (xpp, ypp, zpp), method='linear')
    print q0p.shape
    mlab.contour3d(xpp, ypp, zpp, q0p, 
                   contours=64, transparent=True, colormap='Spectral')
    mlab.show()
    exit()

    q0p[:-1,:] = q[0]
    q0p[-1,:] = q[0,0,:]
    print rp.shape, tp.shape, q0p.shape
    scft_contourf(rp*np.cos(tp), rp*np.sin(tp), q0p)
    plt.xlabel('q[0]')
    plt.show()
    #plt.contourf(r*np.cos(t), r*np.sin(t), w)
    w = np.zeros([Nt, Nz, N2])
    w = np.cos(t)
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
    q_solver = ETDRK4Cylind(R, Nr, Nt, Ns, h=ds, lbc=lbc, rbc=rbc)
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


def cheb_interp3d_r(u, vr):
    '''
    Use chebyshev interpolation for the last dimension of cylindrical
    coordinates (theta, z, r).
    u(theta, z, r): source data, note that the range of r is (0, 1]
    vr: vector to be interpolated, size is Nrp.
    '''
    Nx, Ny, N2 = u.shape
    Nrp = vr.size
    uout = np.zeros([Nx, Ny, Nrp])
    vrp = np.linspace(0, 1, Nrp)
    T = cheb_barycentric_matrix(vrp, 2*N2-1)
    #print Nx, Ny, Nz, Nzp, T.shape, u[0,0].shape
    for i in xrange(Nx):
        for j in xrange(Ny):
            up = u[i,j]
            up = np.hstack((up, up[::-1]))
            uout[i,j] = np.dot(T, up)
    return uout


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
        cmap = plt.cm.Spectral
    # actual plot
    ax.contourf(x, y, z, levels=levels, cmap=cmap, 
                antialiased=False, **kwargs)


if __name__ == '__main__':
    test_etdrk4cylind()

