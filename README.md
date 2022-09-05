## SNN: A lightweight fast exact radius query algorithm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SNN is a fast exact radius neareast neighbor search algorithm. It uses singular value decomposition to reduce the search space and speedup euclidean calculation with BLAS level 2 routine.  SNN enjoys faster speed with trivial python implementation compared to KDtree and Balltree in scikit-learn package. 



### Installation

We open source the C++ code with simple Cmake installing procedure. SNN has dependencies on CBLAS and LAPACK, ensure install them before formally installing SNN. 

```
git clone https://github.com/nla-group/snn.git
cd snn
cmake . # or mkdir build -> cd build -> cmake ../
make 
```

After installation, you can just use 'include "snn.h"' in your code, while compile it simply by linking libsnn.a, CBLAS and LAPACK library. 
For example, you can use g++ by 'g++ your_code.cpp libsnn.a -o output -llapacke -lgslcblas -lm -W'.


### API



