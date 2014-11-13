from distutils.core import setup
from distutils.extension import Extension
# from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_module = Extension(
    "piqmc",
    ["sa.pyx", "qmc.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp']
)

setup(
    name = "piqmc",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
)
