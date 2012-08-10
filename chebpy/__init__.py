# -*- coding: utf-8 -*-
"""
chebpy
======

**chebpy** is a python package for chebyshev series, chebyshev polynomials,
and their related applications in interpolation and spectral methods.

References
----------

* Trefethen, LN *Spectral Methods in Matlab*, 2000, SIAM.
* Kopriva, DA *Implementing Spectral Methods for Partial Differential 
Equations: Algorithms for Scientists and Engineers*, 2008, Springer

"""

__author__ = "Yi-Xin Liu <liuyxpp@gmail.com>"
__license__ = "BSD License"
__version__ = "0.1"

from .cheb import *
from .chebt import *
from .chebd import *
#from .chebp import *
from .cheba import *

