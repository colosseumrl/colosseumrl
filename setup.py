import setuptools
from setuptools import Extension, setup
from Cython.Distutils import build_ext
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

extensions = [
    Extension("colosseumrl.envs.tron.CyTronGrid", ["colosseumrl/envs/tron/CyTronGrid.pyx"],
              extra_compile_args=["-O3", "-march=native"])
]


with open("requirements.txt", 'r') as file:
    requirements = file.readlines()

requirements = list(map(str.strip, requirements))

setup(
    name="colosseumrl",
    version="1.0.0",
    author="Alexander Shmakov",
    author_email="alexanders101@gmail.com",
    description="UC Irvine multi-agent reinforcement learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/colosseumrl/Colosseum",
    packages=setuptools.find_packages(),
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': build_ext},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)