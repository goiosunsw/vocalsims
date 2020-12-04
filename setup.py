import distutils.core
import Cython.Build
import numpy


import unittest
def my_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

distutils.core.setup(
    name = 'vocalsims',
    version = '0.0.2',
    description = 'vocal and lip instrment simulations',
    author = 'Andre Goios',
    license='GPL v3',
    packages=['vocalsims', 'vocalsims.interfacing','vocalsims.fitting',
              'vocalsim_parameters','vocalsim_parameters.vocal_tracts'],
    test_suite = 'setup.my_tests',
    #ext_modules = Cython.Build.cythonize("vocalsims/real_time_filter.pyx"), 
    ext_modules = [distutils.core.Extension('vocalsims.real_time_filter',['vocalsims/real_time_filter.pyx'], 
                                            include_dirs=[numpy.get_include()])],
    include_dirs=[numpy.get_include()])
