#!/usr/bin/env python
"""Setup script for the theta package
"""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy
import os, io, re

extensions = [
    Extension('riemann_theta',
              sources=['riemann_theta.pyx',
                       'finite_sum.c'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-std=c99'],
    ),    
    Extension('radius',
              sources=['radius.pyx',
                       'lll_reduce.c'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-std=c99'],
    ),
    Extension('integer_points',
              sources=['integer_points.pyx'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-std=c99'],
    ),
]


def read(*names, **kwargs):
        with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
        ) as fp:
            return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='openRT',
    description='Open Riemann-Theta',
    version=find_version("__init__.py"),
    author='S. Carrazza, D. Krefl',
    author_email='stefano.carrazza@cern.ch',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    url='https://github.com/RiemannAI/openRT',
    license='LICENSE',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.13",
    ],
    zip_safe=False,
)
