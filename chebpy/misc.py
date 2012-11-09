# -*- coding: utf-8 -*-
"""
chebpy.misc
===========

Misc funcitons.

"""

import numpy as np

from chebpy import EPS, DIRICHLET, NEUMANN, ROBIN

__all__ = ['almost_equal',
           'BC', # class for boundary condition
          ]

class BC(object):
    def __init__(self, kind=None, vc=None):
        '''
        The boundary condition can be generally written as:
            alpha * du/dx + beta * u = gamma
        It is convenient to specified BC by a 3-element vector:
            (alpha, beta, gamma)
        :param:kind: kind of boundary conditions.
        :param:vc: boundary condition specified by a vec with 3 elements.
        '''
        if vc is not None and len(vc) != 3:
            raise ValueError('The vector to specify boundary condtion'
                             'must has 3 elements!')
        if kind == ROBIN and vc is None:
            raise ValueError('Robin BC needs a coefficient vector.')
            
        self.kind = kind
        if kind is None:
            if vc is None:
                self.kind = DIRICHLET
            else:
                self.kind = ROBIN

        if self.kind == DIRICHLET:
            self.alpha = 0
            self.beta = 1.
            self.gamma = 0
        elif self.kind == NEUMANN:
            self.alpha = 1.
            self.beta = 0
            self.gamma = 0
        elif kind == ROBIN:
            self.alpha = vc[0]
            self.beta = vc[1]
            self.gamma = vc[2]
        else:
            raise ValueError('kind ' + str(kind) + ' is not supported.')


def almost_equal(a, b, tol=24):
    '''
    Float comparison for array_like objects.
    
    :param:tol: number of EPS for the tolerence.
    '''

    return np.allclose(a-b, 0, EPS, tol*EPS)


