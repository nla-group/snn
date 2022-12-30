import setuptools
import numpy
from setuptools.command.build_ext import build_ext

with open("README.md", 'r') as f:
    long_description = f.read()
    
setuptools.setup(
    name="snnpy",
    packages=["snnpy"],
    version="0.0.4",
    cmdclass={'build_ext': build_ext},
    setup_requires=["numpy"],
    install_requires=["numpy", "scipy"],
    include_dirs=[numpy.get_include()],
    author="Xinye Chen, Stefan Güttel",
    maintainer="Xinye Chen, Stefan Güttel",
    
    classifiers=["Intended Audience :: Science/Research",
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
