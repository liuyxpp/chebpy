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

from .common import *
from .misc import *
from .tridiag import *
from .chebt import * # Chebyshev transform
from .chebd import * # Chebyshev differentiation
from .cheb import * # Chebyshev polynomials
from .chebi import * # Chebyshev interpolations
from .integral import *
from .chebq import * # Chebyshev quadratures
from .osf import * # OSS and OSC class
from .oscheb import * # OSCHEB class
from .etdrk4 import * # ETDRK4 class, schemes and coefficients
from .cheba import * # Chebyshev applications on solution of PDEs

