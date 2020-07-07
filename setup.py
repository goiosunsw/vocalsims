import distutils.core
import Cython.Build
import numpy
distutils.core.setup(
    ext_modules = Cython.Build.cythonize("real_time_filt.pyx",annotate=True),
    include_dirs=[numpy.get_include()])