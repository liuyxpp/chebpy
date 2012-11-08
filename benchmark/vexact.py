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
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import mpltex.acs

def plot_q():
    data_name = 'exact/ETDRK4_Krogstad_N128_Ns20000'
    
    data_dbc_dbc = data_name
    data_nbc_nbc = 'NBC-NBC/' + data_name
    data_nbc_dbc = 'NBC-DBC/' + data_name
    data_rbc_dbc = 'RBC-DBC/' + data_name
    data_rbc_rbc = 'RBC-RBC/' + data_name

    mat_dbc_dbc = loadmat(data_dbc_dbc)
    mat_nbc_nbc = loadmat(data_nbc_nbc)
    mat_nbc_dbc = loadmat(data_nbc_dbc)
    mat_rbc_dbc = loadmat(data_rbc_dbc)
    mat_rbc_rbc = loadmat(data_rbc_rbc)

    x = mat_dbc_dbc['x']
    q_dbc_dbc = mat_dbc_dbc['q']
    q_nbc_nbc = mat_nbc_nbc['q']
    q_nbc_dbc = mat_nbc_dbc['q']
    q_rbc_dbc = mat_rbc_dbc['q']
    q_rbc_rbc = mat_rbc_rbc['q']

    minor_locator_x = AutoMinorLocator(5)
    minor_locator_y = AutoMinorLocator(5)
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, q_dbc_dbc, 'b-', label='DBC-DBC')
    ax.plot(x, q_nbc_nbc, 'g-', label='NBC-NBC')
    ax.plot(x, q_nbc_dbc, 'r-', label='NBC-DBC')
    ax.plot(x, q_rbc_dbc, 'c-', label='RBC-DBC')
    ax.plot(x, q_rbc_rbc, 'm-', label='RBC-RBC')
    ax.legend(loc='upper right')
    ax.axis([0, 10, 0, 1.15])
    ax.xaxis.set_minor_locator(minor_locator_x)
    ax.yaxis.set_minor_locator(minor_locator_y)
    plt.xlabel('$z$')
    plt.ylabel('$q(z)$')
    fig_name = 'q_exact_all'
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_q()

