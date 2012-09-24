# -*- coding: utf-8 -*-
#/usr/bin/env python
"""
vexact
======

Visualize the exact resolutions calculated by various methods and their relative errors.

"""

from time import time
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import mpltex.acs

def plot_error_Q():
    data_name1 = 'N32_Ns20000'
    data_name2 = 'N32_Ns100000'
    data_name3 = 'N32_Ns1000000'
    data_name4 = 'N64_Ns20000'
    data_name5 = 'N64_Ns100000'
    data_name6 = 'N64_Ns1000000'
    data_name7 = 'N128_Ns20000'
    data_name8 = 'N128_Ns100000'
    data_name9 = 'N128_Ns1000000'
    data_name10 = 'N256_Ns20000'
    data_name11 = 'N256_Ns100000'
    data_name12 = 'N256_Ns1000000'
    data_name13 = 'N2048_Ns20000'
    data_name14 = 'N2048_Ns100000'
    data_name15 = 'N4096_Ns20000'
    data_name16 = 'N4096_Ns100000'
    data_name17 = 'N8192_Ns200000'
    data_name18 = 'N16384_Ns1000000'
    
    data_oss1 = 'OSS_' + data_name7
    data_oss2 = 'OSS_' + data_name9
    data_oss3 = 'OSS_' + data_name10
    data_oss4 = 'OSS_' + data_name12
    data_oss5 = 'OSS_' + data_name15
    data_oss6 = 'OSS_' + data_name16
    data_oss7 = 'OSS_' + data_name18
    data_oscheb1 = 'OSCHEB_' + data_name7
    data_oscheb2 = 'OSCHEB_' + data_name9
    data_oscheb3 = 'OSCHEB_' + data_name10
    data_oscheb4 = 'OSCHEB_' + data_name12
    data_oscheb5 = 'OSCHEB_' + data_name13
    data_oscheb6 = 'OSCHEB_' + data_name14
    data_oscheb7 = 'OSCHEB_' + data_name17
    data_krog1 = 'Krogstad/ETDRK4_' + data_name1
    data_krog2 = 'Krogstad/ETDRK4_' + data_name2
    data_krog3 = 'Krogstad/ETDRK4_' + data_name3
    data_krog4 = 'Krogstad/ETDRK4_' + data_name4
    data_krog5 = 'Krogstad/ETDRK4_' + data_name5
    data_krog6 = 'Krogstad/ETDRK4_' + data_name6
    data_krog7 = 'Krogstad/ETDRK4_' + data_name7
    data_krog8 = 'Krogstad/ETDRK4_' + data_name8
    data_krog9 = 'Krogstad/ETDRK4_' + data_name9
    data_krog10 = 'Krogstad/ETDRK4_' + data_name10
    data_krog11 = 'Krogstad/ETDRK4_' + data_name11
    data_krog12 = 'Krogstad/ETDRK4_' + data_name12
    data_krog13 = 'Krogstad/ETDRK4_' + data_name13
    data_krog14 = 'Krogstad/ETDRK4_' + data_name14

    mat_oss1 = loadmat(data_oss1)
    mat_oss2 = loadmat(data_oss2)
    mat_oss3 = loadmat(data_oss3)
    mat_oss4 = loadmat(data_oss4)
    mat_oss5 = loadmat(data_oss5)
    mat_oss6 = loadmat(data_oss6)
    mat_oss7 = loadmat(data_oss7)
    mat_oscheb1 = loadmat(data_oscheb1)
    mat_oscheb2 = loadmat(data_oscheb2)
    mat_oscheb3 = loadmat(data_oscheb3)
    mat_oscheb4 = loadmat(data_oscheb4)
    mat_oscheb5 = loadmat(data_oscheb5)
    mat_oscheb6 = loadmat(data_oscheb6)
    mat_oscheb7 = loadmat(data_oscheb7)
    mat_krog1 = loadmat(data_krog1)
    mat_krog2 = loadmat(data_krog2)
    mat_krog3 = loadmat(data_krog3)
    mat_krog4 = loadmat(data_krog4)
    mat_krog5 = loadmat(data_krog5)
    mat_krog6 = loadmat(data_krog6)
    mat_krog7 = loadmat(data_krog7)
    mat_krog8 = loadmat(data_krog8)
    mat_krog9 = loadmat(data_krog9)
    mat_krog10 = loadmat(data_krog10)
    mat_krog11 = loadmat(data_krog11)
    mat_krog12 = loadmat(data_krog12)
    mat_krog13 = loadmat(data_krog13)
    mat_krog14 = loadmat(data_krog14)

    Q_oss = np.zeros(7)
    N_oss = np.zeros(7)
    Q_oscheb = np.zeros(7)
    N_oscheb = np.zeros(7)
    Q_krog = np.zeros(14)
    N_krog = np.zeros(14)
    for i in np.arange(7):
        N_oss[i] = i + 1
        Q_oss[i] = eval('mat_oss'+str(i+1)+"['Q']")
        N_oscheb[i] = i + 1
        Q_oscheb[i] = eval('mat_oscheb'+str(i+1)+"['Q']")
    for i in np.arange(14):
        N_krog[i] = i + 1
        Q_krog[i] = eval('mat_krog'+str(i+1)+"['Q']")

    plt.figure()
    #plt.plot(N_oss, Q_oss, 'b^-', label='OSS')
    #plt.plot(N_krog, Q_krog, 'ro-', label='ETDRK4-Krogstad')
    #plt.plot(N_oscheb, Q_oscheb, 'gD-', label='OSCHEB')
    #plt.plot(N_oss, Q_oss, 'b^-', label='OSS')
    plt.plot(N_krog[3:11], Q_krog[3:11] - 4.5626155, 'ro-', label='ETDRK4-Krogstad')
    plt.plot(N_oscheb, Q_oscheb - 4.5626155, 'gD-', label='OSCHEB')
    plt.grid('on')
    plt.legend(loc='upper right')
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.axis([-1, 15, 4.56255, 4.56265])
    plt.xlabel('Parameter set')
    plt.ylabel('Q')
    fig_name = 'range_3_11'
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()


def plot_q():
    pass


if __name__ == '__main__':
    plot_error_Q()
    #plot_q()
