import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

# macOS-specific compiler setup (optional, uncomment if needed)
# os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
# os.environ["CXX"] = "/usr/local/opt/llvm/bin/clang++"

with open("README.md", 'r') as f:
    long_description = f.read()

__version__ = "0.0.9"

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)

# Platform-specific arguments
extra_compile_args = ['-O3']
extra_link_args = ['-lblas']
if os.name == "posix" and "darwin" in os.sys.platform:  # macOS
    extra_compile_args.extend(['-Xpreprocessor', '-fopenmp'])
    extra_link_args.extend(['-lomp', '-L/usr/local/opt/openblas/lib'])
elif os.name == "posix":  # Linux
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')
elif os.name == "nt":  # Windows
    extra_compile_args.append('/openmp')  # MSVC

snn_module = Pybind11Extension(
    'snnpy.snnomp',  
    sources=['snnpy/snnpy.cpp'],
    define_macros=[("VERSION_INFO", __version__)],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language='c++',
)

setuptools.setup(
    name="snnpy",
    packages=["snnpy"],
    version=__version__,
    cmdclass={'build_ext': CustomBuildExtCommand},
    setup_requires=["numpy", "pybind11>=2.2"],
    ext_modules=[snn_module],
    install_requires=["numpy", "scipy", "pybind11>=2.2"],
    author="Xinye Chen, Stefan Güttel",
    maintainer="Xinye Chen, Stefan Güttel",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    author_email="xinye.chen@manchester.ac.uk, stefan.guettel@manchester.ac.uk",
    maintainer_email="xinye.chen@manchester.ac.uk, stefan.guettel@manchester.ac.uk",
    description="A lightweight fast exact radius query algorithm",
    long_description_content_type='text/markdown',
    long_description=long_description,
    url="https://github.com/nla-group/snn.git",
    license='MIT License'
)
