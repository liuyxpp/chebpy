# -*- coding: utf-8 -*-
"""
chebpy.misc
===========

Misc funcitons.

"""

import numpy as np
import sys
EPS = sys.float_info.epsilon

__all__ = ['almost_equal',
           'EPS',
          ]

def almost_equal(a, b, tol=24):
    '''
    Float comparison for array_like objects.
    
    :param:tol: number of EPS for the tolerence.
    '''

    return np.allclose(a-b, 0, EPS, tol*EPS)


