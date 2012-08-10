# -*- coding: utf-8 -*-
"""
chebpy.cheb
===========

Chebyshev polynomial.

"""

import numpy as np
from math import cos, acos

__all__ = ['cheb_polynomial_recursion',
           'cheb_polynomial_trigonometric',
           'cheb_polynomial_series']

def cheb_polynomial_recursion(k, x):
    '''
    Compute T_k(x) = cos[k*arccos(x)] via recursion method.
    The recursive formula is (for k>1)
        T_k+1(x) = 2*x*T_k(x) - T_k-1(x)
    with
        T_0 = 1; T_1 = x

    Quoted from Kopriva, DA 2009:
        For |x| > 1/2, the coefficient of the leading order polynomial
        is larger than 1. This makes the algorithm (recursion) unstable
        for large k since (rounding) errors in the lower order
        polynomials are amplified.
    Thus, it is wise to switch to the cheb_polynomial_trigonometric
    as k > Kc, where Kc is around 70.

    :param:k: the order for the chebyshev series term.
    :param:x: the variable.
    '''

    if k == 0:
        return 1
    if k == 1:
        return x

    T0 = 1
    T1 = x
    for j in xrange(2,k+1):
        Tk = 2. * x * T1 - T0
        T0 = T1
        T1 = Tk
    return Tk


def cheb_polynomial_trigonometric(k, x):
    '''
    Compute T_k(x) = cos[k*arccos(x)] directly.

    :param:k: the order for the chebyshev series term.
    :param:x: the variable.
    '''

    if k == 0:
        return 1
    if k == 1:
        return x

    return cos(k * acos(x))


def cheb_polynomial_series(k, x):
    '''
    Compute T_0, T_1, ..., T_k via recursion method.
    The recursive formula is (for k>1)
        T_k+1(x) = 2*x*T_k(x) - T_k-1(x)
    with
        T_0 = 1; T_1 = x

    :param:k: the order for the chebyshev series term.
    :param:x: the variable.
    '''

    if k == 0:
        return np.array([1])
    if k == 1:
        return np.array([1, x])

    T0 = 1
    T1 = x
    TT = [T0, T1]
    for j in xrange(2, k+1):
        Tk = 2. * x * T1 - T0
        TT.append(Tk)
        T0 = T1
        T1 = Tk
    return np.array(TT)


