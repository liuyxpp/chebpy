# -*- coding: utf-8 -*-
#/usr/bin/env python

import numpy as np
import matplotlib.pylab as plt

from chebpy import cheb_D2_mat_robin_robin, cheb_D2_mat
from chebpy import cheb_D2_mat_robin_robin_1
from chebpy import BC, ROBIN, DIRICHLET, NEUMANN


def test_robin_robin():
    N = 4
    ka = 1
    kb = -1
    D1t, D2t, x = cheb_D2_mat_robin_robin(N, ka, kb)
    D2t_2, x = cheb_D2_mat_robin_robin_1(N, ka, kb)
    D1, D2, x = cheb_D2_mat(N)
    #print x
    #print D2
    print D2t
    print D2t_2
    #print D1
    #print D1t


if __name__ == '__main__':
    test_robin_robin()
