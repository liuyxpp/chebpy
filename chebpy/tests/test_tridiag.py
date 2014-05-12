# -*- coding: utf-8 -*-
"""
chebpy.tests.test_tridiag
=========================

Test tridiag.py

"""

import unittest
import numpy as np
from chebpy import almost_equal
from chebpy import solve_tridiag_thual, solve_tridiag_complex_thual

class TridiagTest(unittest.TestCase):

    def testThual(self):
        L = np.array([[1,1,1,1,1,1],
                      [3,2,1,0,0,0],
                      [0,6,5,4,0,0],
                      [0,0,7,9,8,0],
                      [0,0,0,3,1,0],
                      [0,0,0,0,2,7]])
        f = np.array([0,3,5,8,7,4])
        u0 = np.linalg.solve(L,f)

        p = np.array([3,6,7,3,2])
        q = np.array([2,5,9,1,7])
        r = np.array([1,4,8,0,0])
        c = np.array([1,1,1,1,1,1])

        u = solve_tridiag_thual(p,q,r,c,f)

        self.assertTrue(almost_equal(u0,u))
        
    def testThualComplex(self):
        L = np.array([[1,1,1,1,1,1],
                      [3+1j,2,1,0,0,0],
                      [0,6,5,4-1j,0,0],
                      [0,0,7,9,8,0],
                      [0,0,0,3,1,0],
                      [0,0,0,0,2,7]])
        f = np.array([0,3,5,8,7,4])
        u0 = np.linalg.solve(L,f)

        p = np.array([3+1j,6,7,3,2])
        q = np.array([2,5,9,1,7])
        r = np.array([1,4-1j,8,0,0])
        c = np.array([1,1,1,1,1,1])

        u = solve_tridiag_complex_thual(p,q,r,c,f)

        # it will fail if tol < 40
        self.assertTrue(almost_equal(u0.real, u.real, 40))
        self.assertTrue(almost_equal(u0.imag, u.imag, 40))
        
