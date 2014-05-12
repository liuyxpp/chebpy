# -*- coding: utf-8 -*-
"""
chebpy.tests.test_integral
==========================

Test integral.py

"""

import unittest
import numpy as np
from chebpy import almost_equal
from chebpy import oss_integral
from scipy.integrate import simps, romb

class IntegralTest(unittest.TestCase):

    def testOSS(self):
        #I0 = 1. / 6
        I0 = 2.5 - np.e
        err0 = 1.
        for n in np.power(2, np.arange(2,16)):
            N = int(n)
            x = 1. * np.arange(N+1) / N
            #f = 0.25 - (x - 0.5)**2
            f = (np.exp(x) - 1) * (x - 1)
            I = oss_integral(f)
            #I = simps(f, dx=1./N)
            #I = romb(f, dx=1./N)
            err1 = np.abs(I - I0) / np.abs(I0)
            print N, '\t', err1, '\t', err1/err0, '\t', I
            err0 = err1
        #self.assertTrue(almost_equal(I0, I))
        self.assertTrue(False)
        
