'''
chebpy
======

**chebpy** is a python package for spetral methods of PDEs based on
Chebyshev seriers. 

Quickstart
----------

1. Install
^^^^^^^^^^

::

    $ easy_install gyroid

or

::

    $ tar -xvf chebpy-xxx.tar.gz
    $ cd chebpy-xxx
    $ python setup.py install

Required packages:

* `numpy`: it should be installed before installing gyroid.
* `scipy`: use it to save data in Matlab mat format.

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

Ask for Help
'''
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='chebpy',
    version='0.1',
    license='BSD',
    description='Chebyshev polynomial based spectral methods of PDEs.',
    author='Yi-Xin Liu',
    author_email='liuyxpp@gmail.com',
    url='https://bitbucket.org/liuyxpp/chebpy',
    packages=['chebpy'],
    include_package_data=True,
    zip_safe=False,
    long_description=__doc__,
    platform='linux',
    install_requires=[
        'numpy',
        'scipy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Education',
    ]
     )

