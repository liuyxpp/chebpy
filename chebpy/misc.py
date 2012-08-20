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

def almost_equal(a, b):
    return np.allclose(a-b, 0, EPS, 24.*EPS)


