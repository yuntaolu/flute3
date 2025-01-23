from setuptools import setup, Extension
import pybind11
from cmake_build_extension import BuildExtension, CMakeExtension

setup(
    name="pyflute",
    version="0.1",
    author="YT",
    description="Python bindings for flute3",
    ext_modules=[CMakeExtension(name="pyflute")],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
