# python/setup.py
import os
import sys
from setuptools import setup, Extension
import pybind11
import numpy as np


# Get the absolute path to the Flute source directory
FLUTE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ext_modules = [
    Extension(
        "pyflute._pyflute",
        ["pyflute/_pyflute.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            FLUTE_ROOT,  # Include main Flute directory
            np.get_include(),
        ],
        libraries=['flute'],
        library_dirs=[os.path.join(FLUTE_ROOT, 'build')],
        runtime_library_dirs=[os.path.join(FLUTE_ROOT, 'build')],
        extra_compile_args=['-std=c++11', '-fPIC'],
        language='c++'
    ),
]

setup(
    name="pyflute",
    version="2.12.1",
    author="Pineapple",
    author_email="Pineapple@fake.com",  
    description="Python bindings for FLUTE",
    long_description="Python bindings for FLUTE - Fast Look-Up Table Based Rectilinear Steiner Minimal Tree Algorithm",
    packages=['pyflute'],
    package_dir={'': '.'},
    ext_modules=ext_modules,
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.15.0',
        'pybind11>=2.6.0',
    ],
    zip_safe=False,
)