from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[ Extension("fast_digamma",
              ["fast_digamma.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "fast_digamma",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)