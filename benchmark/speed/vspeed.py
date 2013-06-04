#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vspeed
======

A script for visulizing results of speed test of OSS, OSCHEB, and ETDRK4.

Copyright (C) 2012 Yi-Xin Liu

"""

import argparse
import os.path

import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import mpltex.acs
from scftpy import Brush, SCFTConfig

parser = argparse.ArgumentParser(description='vbrush options')
parser.add_argument('-c', '--config',
                    default='param.ini',
                    help='the configuration file of polyorder.')

args = parser.parse_args()

def plot_t_N():
    dirlist = ['speed_OSS_N', 'speed_OSCHEB_N', 
               'speed_ETDRK4_N',]
    fig_name = 't_N'

    plt.figure()
    ax = plt.subplot(111)

    mat_oss = loadmat('speed_OSS_N')
    N_oss = mat_oss['N']
    t_oss = mat_oss['t']
    tf_oss = mat_oss['t_full']
    ax.plot(N_oss, tf_oss, 'bv:', mew=.2, mfc='w', mec='b', 
            label='OSS full')
    ax.plot(N_oss, t_oss, 'bv-', mew=0, label='OSS core')

    mat_oscheb = loadmat('speed_OSCHEB_N')
    N_oscheb = mat_oscheb['N']
    t_oscheb = mat_oscheb['t']
    tf_oscheb = mat_oscheb['t_full']
    ax.plot(N_oscheb, tf_oscheb, 'g^:', mew=.2, mfc='w', mec='g', 
            label='OSCHEB full')
    ax.plot(N_oscheb, t_oscheb, 'g^-', mew=0, label='OSCHEB core')

    mat_etdrk4 = loadmat('speed_ETDRK4_N')
    N_etdrk4 = mat_etdrk4['N']
    t_etdrk4 = mat_etdrk4['t']
    tf_etdrk4 = mat_etdrk4['t_full']
    ax.plot(N_etdrk4, tf_etdrk4, 'ro:', mew=.2, mfc='w', mec='r', 
            label='ETDRK4 full')
    ax.plot(N_etdrk4, t_etdrk4, 'ro-', mew=0, label='ETDRK4 core')

    plt.xlabel('$N_z$')
    plt.ylabel('Computation time')
    plt.xscale('log')
    plt.yscale('log')
    plt.axis([0.3, 1000000, 6e-3, 1000])
    ax.legend(loc='upper left')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()


def plot_t_err():
    dirlist = ['speed_OSS_N', 'speed_OSCHEB_N', 'speed_ETDRK4_N', 
               'speed_OSS_accuracy', 'speed_OSCHEB_accuracy',
               'speed_OSCHEB_accuracy_Ns20000']
    fig_name = 't_err'

    plt.figure()
    ax = plt.subplot(111)

    mat_oss = loadmat('speed_OSS_N')
    t_oss = mat_oss['t']
    err_oss = mat_oss['err']
    ax.plot(t_oss, err_oss, 'bv-', mew=0, label='OSS \n $0.005$')
    mat_oss = loadmat('speed_OSS_accuracy')
    t_oss = mat_oss['t']
    err_oss = mat_oss['err']
    ax.plot(t_oss, err_oss, 'bv:', mew=0.2, mfc='w', mec='b',
            label='OSS \n $5\\times 10^{-5}$')

    mat_oscheb = loadmat('speed_OSCHEB_N')
    t_oscheb = mat_oscheb['t']
    err_oscheb = mat_oscheb['err']
    ax.plot(t_oscheb, err_oscheb, 'g^-', mew=0, label='OSCHEB \n $0.005$')
    mat_oscheb = loadmat('speed_OSCHEB_accuracy_Ns20000')
    t_oscheb = mat_oscheb['t']
    err_oscheb = mat_oscheb['err']
    ax.plot(t_oscheb, err_oscheb, 'g^:', mew=0.2, mfc='w', mec='g', 
            label='OSCHEB \n $5 \\times 10^{-5}$')
    #mat_oscheb = loadmat('speed_OSCHEB_accuracy')
    #t_oscheb = mat_oscheb['t']
    #err_oscheb = mat_oscheb['err']
    #ax.plot(t_oscheb, err_oscheb, 'g^-.', mew=0.2, mfc='w', mec='g', 
    #        label='OSCHEB $\Delta s=10^{-5}$')

    mat_etdrk4 = loadmat('speed_ETDRK4_N')
    t_etdrk4 = mat_etdrk4['t']
    err_etdrk4 = mat_etdrk4['err']
    ax.plot(t_etdrk4, err_etdrk4, 'ro-', mew=0, label='ETDRK4 \n $0.005$')

    plt.xlabel('Computation time')
    plt.ylabel('Relative error in $Q$')
    plt.xscale('log')
    plt.yscale('log')
    plt.axis([1e-3, 1e5, 1e-12, 100])
    ax.legend(loc='upper right')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    #plot_t_N()
    plot_t_err()


