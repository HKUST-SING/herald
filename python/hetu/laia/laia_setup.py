from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy 
## define the extension module 
ext_module = Extension('laia', sources=['laia.pyx'], 
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    language="c++") 

## run the setup 
setup(name='laia', ext_modules=cythonize(ext_module)) 
