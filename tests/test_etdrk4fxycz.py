# -*- coding: utf-8 -*-
#/usr/bin/env python

import numpy as np
#import matplotlib.pylab as plt
#from mayavi import mlab

import mpltex.acs
from chebpy import ETDRK4FxyCz, BC, ETDRK4
from chebpy import ROBIN, DIRICHLET, cheb_barycentric_matrix
from timer import Timer

def test_etdrk4fxycz():
    '''
    Assume the following PDE in cylindrical coordinates,
            du/dt = (d^2/dx^2 + d^2/dy^2 + d^2/dz^2) u - w u
    with u = u(x,y,z), w=w(x,y,z) in the domain
            [0, Lx] x [0, Ly] x [0, Lz]
    for time t=0 to t=1,
    with boundary conditions
            u(x,y,z,t) = u(x+Lx,y,z,t) # periodic in x direction
            u(x,y,z,t) = u(x,y+Ly,z,t) # periodic in y direction
            d/dr[u(z=0,y,z,t)] = ka u(z=0)
            d/dr[u(z=Lz,y,z,t)] = kb u(z=Lz)
    Test: PASSED 2013.8.16.
    '''
    Lx = 2.0*np.pi # x [0, Lx]
    Nx = 32
    Ly = 2.0*np.pi # y [0, Ly]
    Ny = 32
    Lz = 3.0 # z [0, Lz]
    Nz = 63
    Np = 64
    Ns = 101
    ds = 1. / (Ns - 1)
    show3d = False

    # Periodic in x direction, Fourier
    xx = np.arange(Nx) * Lx / Nx
    yy = np.arange(Ny) * Ly / Ny
    # Non-periodic in z direction, Chebyshev
    ii = np.arange(Nz+1)
    zz = np.cos(np.pi * ii / Nz) # zz [-1, 1]
    zz = 0.5 * (zz + 1) * Lz # mapping to [0, Lz]
    zzp = np.linspace(0, Lz, Np)
    x, y, z = np.meshgrid(xx, yy, zz)
    #xp, yp, zp = np.meshgrid(xx, yy, zzp)
    xp, yp, zp = np.mgrid[0:Lx*(1-1./Nx):Nx*1j,
                          0:Ly*(1-1./Ny):Ny*1j,
                          0:Lz:Np*1j]
    A = 1.0
    w = np.sin(x + 2*y) + np.cos(z)**2
    q = np.zeros([Ns, Nx, Ny, Nz+1])
    q[0] = np.exp(np.sin(x*y) - A*z)
    q0p = np.exp(np.sin(xp*yp) - A*zp) # the exact

    #q0 = q[0]
    #plt.imshow(q0[Nx/2,:,:])
    #plt.show()
    #plt.imshow(q0[:,Ny/2,:])
    #plt.show()
    #plt.imshow(q0[:,:,Nz/2])
    #plt.show()
    if show3d:
        q0p2 = cheb_interp3d_z(q[0], zzp)
        print np.mean(np.abs(q0p2 - q0p)) # the error of interpolation
        #plt.plot(zz, q[0, Nx/2, Ny/2], '.')
        #plt.plot(zzp, q0p[Nx/2, Ny/2])
        #plt.plot(zzp, q0p2[Nx/2, Ny/2], '^')
        #plt.show()
        mlab.clf()
        #s = mlab.pipeline.scalar_field(xp,yp,zp,q0p2)
        #mlab.pipeline.iso_surface(s,
                                  #contours=[q0p.min()+0.1*q0p.ptp(),],
        #                          opacity=0.3)
        mlab.contour3d(xp, yp, zp, q0p2,
                       contours=16, opacity = 0.5,
                       transparent=True, colormap='Spectral')
        mlab.show()

    #lbc = BC(DIRICHLET, [0.0, 1.0, 0.0])
    #rbc = BC(DIRICHLET, [0.0, 1.0, 0.0])
    lbc = BC(ROBIN, [1.0, 0.1, 0.0])
    rbc = BC(ROBIN, [1.0, -0.1, 0.0])
    q_solver = ETDRK4FxyCz(Lx, Ly, Lz, Nx, Ny, Nz, Ns, h=ds, lbc=lbc, rbc=rbc)
    print 'Initiate done!'
    M = 10;  # 10 runs took 131.901 seconds for cpu one core (32x32x32)
    with Timer() as t:
        for m in xrange(M):
            q1 = q_solver.solve(w, q[0], q)
    print "10 runs took", t.secs, " seconds."
    print 'Solve done!'

    #plt.imshow(q1[Nx/2,:,:])
    #plt.show()
    #plt.imshow(q1[:,Ny/2,:])
    #plt.show()
    #plt.imshow(q1[:,:,Nz/2])
    #plt.show()
    if show3d:
        q1p = cheb_interp3d_z(q1, zzp)
        mlab.clf()
        mlab.contour3d(xp,yp,zp,q1p,contours=16,transparent=True)
        #mlab.contour3d(q1)
        mlab.show()


def cheb_interp3d_z(u, vz):
    '''
    u(x,y,z): source data
    vz: vector to be interpolated, size is Nzp
    '''
    Nx, Ny, Nz = u.shape
    Nzp = vz.size
    uout = np.zeros([Nx, Ny, Nzp])
    vzp = np.linspace(-1, 1, Nzp)
    T = cheb_barycentric_matrix(vzp, Nz-1)
    #print Nx, Ny, Nz, Nzp, T.shape, u[0,0].shape
    for i in xrange(Nx):
        for j in xrange(Ny):
            uout[i,j] = np.dot(T, u[i,j])
    return uout


if __name__ == '__main__':
    test_etdrk4fxycz()

