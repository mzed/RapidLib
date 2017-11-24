from distutils.core import setup, Extension
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

src_dir = '../src/'

# bayesfilter extension module
bayesfilter_module = Extension('_bayesfilter',
                           extra_compile_args = [],
                           sources=['bayesfilter_wrap.cxx', src_dir + 'filter_utilities.cpp', src_dir + 'BayesianFilter.cpp'],
                           include_dirs = [numpy_include, src_dir]
                           )
# mhmm_lib setup
setup (name        = 'bayesfilter',
       version     = '0.1',
       author      = "Jules Francoise <jules.francoise@ircam.fr>",
       description = """Simple Bayesian Filter for EMG Envelope extraction""",
       ext_modules = [bayesfilter_module],
       py_modules  = ["bayesfilter"],
       )