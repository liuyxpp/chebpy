chebpy
======

**chebpy** is a python package for spetral methods of PDEs based on
Chebyshev series. 

Quickstart
----------

1. Install
^^^^^^^^^^

::

    $ easy_install chebpy

or

::

    $ tar -xvf chebpy-xxx.tar.gz
    $ cd chebpy-xxx
    $ python setup.py install

Required packages:

* `numpy`: chebpy heavily depends on numpy.
* `scipy`: advanced algorithms, such as scipy.fftpack.dct.

2. APIs
^^^^^^^^
Current available functions:

**Chebyshev series construction**

* cheb_polynomial_recursion
* cheb_polynomial_trigonometric
* cheb_polynomial_series

**Fast Chebyshev transform**

* cheb_fast_transform
* cheb_inverse_fast_transform

**Chebyshev differentiation**

* cheb_D1_mat
* cheb_D1_fft
* cheb_D1_dct
* cheb_D1_fchebt

**Chebyshev interpolation**

* cheb_interpolation_point
* cheb_interpolation_1d
* cheb_interpolation_2d

**Chebyshev quadrature**

* cheb_quadrature_clencurt
* cheb_quadrature_cgl

**Chebyshev applications in solution of PDEs**

* cheb_mde_split
* cheb_mde_neumann_split
* cheb_mde_etdrk4
* cheb_mde_neumann_etdrk4
* cheb_mde_robin_etdrk4
* cheb_allen_cahn_etdrk4

Ask for Help
------------

* You can directly contact me at liuyxpp@gmail.com.
* You can join the mailinglist by sending an email to chebpy@librelist.com 
  and replying to the confirmation mail. 
  To unsubscribe, send a mail to chebpy-unsubscribe@librelist.com 
  and reply to the confirmation mail.

Links
-----

* `Documentation <http://pypi.python.org/pypi/chebpy>`_
* `Website <http://ngpy.org>`_
* `Development version <http://bitbucket.org/liuyxpp/chebpy/>`_

