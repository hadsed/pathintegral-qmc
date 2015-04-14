from setuptools import setup, find_packages
# from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

extensions = [
    Extension(
        "piqmc.sa", ["piqmc/sa.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
        ),
    Extension(
        "piqmc.qmc", ["piqmc/qmc.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
        ),
    Extension(
        "piqmc.tools", ["piqmc/tools.pyx"],
        include_dirs=[numpy.get_include()]
        )
    ]

setup(
    name = "piqmc",
    description="Path-integral quantum Monte Carlo code for simulating quantum annealing.",
    author="Hadayat Seddiqi",
    author_email="hadsed@gmail.com",
    url="https://github.com/hadsed/pathintegral-qmc",
    packages=find_packages(exclude=['testing', 'examples']),
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions,
)
