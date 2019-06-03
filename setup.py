import setuptools
import sys
from setuptools import Extension, setup

# Readme autoload
with open("README.md", "r") as fh:
    long_description = fh.read()

# Requirements autoload
with open("requirements.txt", 'r') as file:
    requirements = file.readlines()

requirements = list(map(str.strip, requirements))

# Cython setup
try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

if '--without-cython' in sys.argv:
    use_cython = False
    sys.argv.remove('--without-cython')

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += [
        Extension("colosseumrl.envs.tron.CyTronGrid", [ "colosseumrl/envs/tron/CyTronGrid.pyx" ]),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("colosseumrl.envs.tron.CyTronGrid", [ "colosseumrl/envs/tron/CyTronGrid.c" ]),
    ]

setup(
    name="colosseumrl",
    version="1.0.4",
    author="Alexander Shmakov",
    author_email="alexanders101@gmail.com",
    description="UC Irvine multi-agent reinforcement learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/colosseumrl/Colosseum",
    packages=setuptools.find_packages(),
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
