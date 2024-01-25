import numpy
from setuptools import setup
from Cython.Build import cythonize

_version = '0.0.1'

with open("README.rst", 'r') as f:
    long_description = f.read()

setup_args = {'name':"dcnn",
        'packages':["dcnn"],
        'version':_version,
        'install_requires':["numpy>=1.3.0", "scipy>=0.7.0"],
        'package_data':{"dcnn": ["native/native_cc.pyx", "snn/snn_cc.pyx"]
                    },
        'classifiers':["Intended Audience :: Science/Research",
                "Intended Audience :: Developers",
                "Programming Language :: Python",
                "Topic :: Software Development",
                "Topic :: Scientific/Engineering",
                'Operating System :: Microsoft :: Windows',
                'Operating System :: POSIX',
                'Operating System :: Unix',
                'Operating System :: MacOS',
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.6",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.10"
                ],
        'long_description':long_description,
        'author':"Xinye Chen, Stefan Güttel",
        'author_email':"xinye.chen@manchester.ac.uk, stefan.guettel@manchester.ac.uk",
        'description':"Fast and explainable clustering based on sorting",
        'long_description_content_type':'text/x-rst',
        'url':"https://github.com/nla-group/CLASSIX.git",
        'license':'MIT License'
}


setup(
    setup_requires=["cython", "numpy>=1.3.0"],
    ext_modules = cythonize(["dcnn/native/native_cc.pyx", "dcnn/snn/snn_cc.pyx"], 
                            include_path=["dcnn"],
                            language="c++"), 

    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    **setup_args
)
