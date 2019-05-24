from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

extensions = [
    Extension("CyTronGrid", ["CyTronGrid.pyx"],
              extra_compile_args=["-O3", "-march=native"])
]

setup(ext_modules=cythonize(extensions, annotate=True), cmdclass={'build_ext': build_ext})
