# -*- coding: utf-8 -*-
"""
chebpy.cheba
============

Applications of Chebyshev spectral methods to solve PDEs.

"""

import numpy as np
from math import cos, acos

__all__ = ['cheb_mde_splitting_pseudospectral',
          ]

def cheb_mde_splitting_pseudospectral():
    '''
    Solve a modified diffusion equation (MDE) via Strang operator splitting
    as time-stepping scheme and pseudospectral methods on Chebyshev grids.

    The MDE is:
        dq/dt = Dq + wq
    where D is Laplace operator.
    '''
    pass


