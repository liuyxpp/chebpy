# -*- coding: utf-8 -*-
#/usr/bin/env python
"""
verror
======

Visualize the benchmark results.

"""

from time import time
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import mpltex.acs

def plot_error_N():
    data_path = 'N_convergence/'
    data_name1 = 'exact_OSCHEB_N8192_Ns200000/N_DBC_Ns20000_hyperbolic'
    data_name2 = 'exact_OSCHEB_N8192_Ns200000/N_DBC_Ns100000_hyperbolic'
    data_oss1 = 'OSS/' + data_name1
    data_oss2 = 'OSS/' + data_name2
    data_oscheb1 = 'OSCHEB/' + data_name1
    data_oscheb2 = 'OSCHEB/' + data_name2
    data_cox1 = 'Cox_Matthews/' + data_name1
    data_cox2 = 'Cox_Matthews/' + data_name2
    data_krog1 = 'Krogstad/' + data_name1
    data_krog2 = 'Krogstad/' + data_name2

    mat_oss1 = loadmat(data_path + data_oss1)
    mat_oss2 = loadmat(data_path + data_oss2)
    mat_oscheb1 = loadmat(data_path + data_oscheb1)
    mat_oscheb2 = loadmat(data_path + data_oscheb2)
    mat_cox1 = loadmat(data_path + data_cox1)
    mat_cox2 = loadmat(data_path + data_cox2)
    mat_krog1 = loadmat(data_path + data_krog1)
    mat_krog2 = loadmat(data_path + data_krog2)

    N11 = mat_oss1['N1']
    N12 = mat_oss2['N1']
    N21 = mat_oscheb1['N2']
    N22 = mat_oscheb2['N2']
    N31 = mat_cox1['N3']
    N32 = mat_cox2['N3']
    N41 = mat_krog1['N3']
    N42 = mat_krog2['N3']
    err11 = mat_oss1['err1']
    err12 = mat_oss2['err1']
    err21 = mat_oscheb1['err2']
    err22 = mat_oscheb2['err2']
    err31 = mat_cox1['err3']
    err32 = mat_cox2['err3']
    err41 = mat_krog1['err3']
    err42 = mat_krog2['err3']

    plt.figure()
    #plt.plot(N11, err11, 'bv-', label='OSS $\Delta s=5e-5$')
    #plt.plot(N11, err11, 'bv:', mew=.2, mfc='w', mec='b', 
    #         label='OSS $\Delta s=1e-5$')
    plt.plot(N11[:5], err11[:5], 'bv-', label='OSS $\Delta s=5e-5$')
    plt.plot(N11[:5], err11[:5], 'bv:', mew=.2, mfc='w', mec='b', 
             label='OSS $\Delta s=1e-5$')
    #plt.plot(N21, err21, 'g^-', label='OSCHEB $\Delta s=5e-5$')
    #plt.plot(N22, err22, 'g^:', mew=.2, mfc='w', mec='g',
    #         label='OSCHEB $\Delta s=1e-5$')
    plt.plot(N21[:5], err21[:5], 'g^-', label='OSCHEB $\Delta s=5e-5$')
    plt.plot(N22[:5], err22[:5], 'g^:', mew=.2, mfc='w', mec='g',
             label='OSCHEB $\Delta s=1e-5$')
    plt.plot(N31, err31, 'mD-', label='ETDRK4-Cox-Matthews $\Delta s=5e-5$')
    plt.plot(N32, err32, 'mD:', mew=.2, mfc='w', mec='m',
             label='ETDRK4-Cox-Matthews $\Delta s=1e-5$')
    plt.plot(N41, err41, 'ro-', label='ETDRK4-Krogstad $\Delta s=5e-5$')
    plt.plot(N42, err42, 'ro:', mew=.2, mfc='w', mec='r',
             label='ETDRK4-Krogstad $\Delta s=1e-5$')
    #plt.axis([1, 300, 1e-13, 1])
    plt.grid('on')
    plt.legend(loc='upper right')
    #plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$N_z$')
    plt.ylabel('Relative error in Q')
    fig_name = 'analysis/exact_OSCHEB_N8192_Ns200000/N_DBC_Ns_both'
    plt.savefig(data_path + fig_name, bbox_inches='tight')
    plt.show()


def plot_error_Ns():
    data_path = 'Ns_convergence/'
    data_name = 'exact_OSCHEB_N8192_Ns200000/Ns_DBC_N256'
    data_oss = 'OSS/' + data_name
    data_oscheb = 'OSCHEB/' + data_name
    data_cox = 'Cox_Matthews/' + data_name
    data_krog = 'Krogstad/' + data_name

    mat_oss = loadmat(data_path + data_oss)
    mat_oscheb = loadmat(data_path + data_oscheb)
    mat_cox = loadmat(data_path + data_cox)
    mat_krog = loadmat(data_path + data_krog)

    Ns1 = mat_oss['Ns1_1']
    Ns2 = mat_oscheb['Ns1_2']
    Ns3 = mat_cox['Ns1_3']
    Ns4 = mat_krog['Ns1_3']
    err1 = mat_oss['err1_1']
    err2 = mat_oscheb['err1_2']
    err3 = mat_cox['err1_3']
    err4 = mat_krog['err1_3']

    plt.figure()
    plt.plot(Ns1, err1, 'bv-', label='OSS')
    plt.plot(Ns2, err2, 'g^-', label='OSCHEB')
    plt.plot(Ns3, err3, 'mD-', label='ETDRK4-Cox-Matthews')
    plt.plot(Ns4, err4, 'ro-', label='ETDRK4-Krogstad')
    #plt.plot(Ns3[:-1], err3[:-1], 'mD-', label='ETDRK4-Cox-Matthews')
    #plt.plot(Ns4[:-1], err4[:-1], 'ro-', label='ETDRK4-Krogstad')
    plt.grid('on')
    plt.axis([9e-6, 1.1, 1e-11, 10])
    #plt.legend(loc='lower right')
    plt.legend(loc='upper left')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\Delta s$')
    plt.ylabel('Relative error in Q')
    fig_name = data_path + 'analysis/' + data_name
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()


def plot_error_Ns_N():
    data_path = 'Ns_convergence/'
    data_name = 'Krogstad/exact_OSCHEB_N8192_Ns200000/Ns_DBC_N'
    data_32 = data_path + data_name + '32'
    data_64 = data_path + data_name + '64'
    data_96 = data_path + data_name + '96'
    data_100 = data_path + data_name + '100'
    data_128 = data_path + data_name + '128'
    data_160 = data_path + data_name + '160'
    data_192 = data_path + data_name + '192'
    data_224 = data_path + data_name + '224'
    data_256 = data_path + data_name + '256'

    mat_krog_32 = loadmat(data_32)
    mat_krog_64 = loadmat(data_64)
    mat_krog_96 = loadmat(data_96)
    mat_krog_100 = loadmat(data_100)
    mat_krog_128 = loadmat(data_128)
    mat_krog_160 = loadmat(data_160)
    mat_krog_192 = loadmat(data_192)
    mat_krog_224 = loadmat(data_224)
    mat_krog_256 = loadmat(data_256)

    Ns_32 = mat_krog_32['Ns1_3']
    Ns_64 = mat_krog_64['Ns1_3']
    Ns_96 = mat_krog_96['Ns1_3']
    Ns_100 = mat_krog_100['Ns1_3']
    Ns_128 = mat_krog_128['Ns1_3']
    Ns_160 = mat_krog_160['Ns1_3']
    Ns_192 = mat_krog_192['Ns1_3']
    Ns_224 = mat_krog_224['Ns1_3']
    Ns_256 = mat_krog_256['Ns1_3']
    err_32 = mat_krog_32['err1_3']
    err_64 = mat_krog_64['err1_3']
    err_96 = mat_krog_96['err1_3']
    err_100 = mat_krog_100['err1_3']
    err_128 = mat_krog_128['err1_3']
    err_160 = mat_krog_160['err1_3']
    err_192 = mat_krog_192['err1_3']
    err_224 = mat_krog_224['err1_3']
    err_256 = mat_krog_256['err1_3']

    plt.figure()
    plt.plot(Ns_32, err_32, 'bv-', label='$N = 32$')
    plt.plot(Ns_64, err_64, 'g^-', label='$N = 64$')
    plt.plot(Ns_96, err_96, color= '#908000', linestyle='-', marker='p', label='$N = 96$')
    plt.plot(Ns_100, err_100, color='#D02030', linestyle='-', marker='s', label='$N = 100$')
    plt.plot(Ns_128, err_128, 'kD-', label='$N = 128$')
    plt.plot(Ns_160, err_160, 'y*-', label='$N = 160$')
    plt.plot(Ns_192, err_192, 'c>-', label='$N = 192$')
    plt.plot(Ns_224, err_224, 'm<-', label='$N = 224$')
    plt.plot(Ns_256, err_256, 'ro-', label='$N = 256$')
    plt.grid('on')
    plt.axis([5e-6, 10., 1e-14, 0.1])
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\Delta s$')
    plt.ylabel('Relative error in Q')
    fig_name = data_path + data_name + '_all'
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()


def plot_q():
    pass


if __name__ == '__main__':
    #plot_error_N()
    #plot_error_Ns()
    plot_error_Ns_N()
    #plot_q()
    #plot_w()
    #plot_u0()
