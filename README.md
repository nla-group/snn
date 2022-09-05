## SNN: A lightweight fast exact radius query algorithm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SNN is a fast exact radius neareast neighbor search algorithm. It uses singular value decomposition to reduce the search space and speedup euclidean calculation with BLAS level 2 routine.  SNN enjoys faster speed with trivial python implementation compared to KDtree and Balltree in scikit-learn package. 



### Installation

We open source the C++ code with simple Cmake installing procedure. SNN has dependencies on CBLAS and LAPACK, ensure install them before formally installing SNN. LAPACK is [available from GitHub](https://github.com/Reference-LAPACK/lapack). LAPACK releases are also [available on netlib](http://www.netlib.org/lapack/).

Install SNN simply by, please modify the ``CMakeList.txt`` file according to your LAPACK location.  
```sh
git clone https://github.com/nla-group/snn.git
cd snn
cmake . # or mkdir build -> cd build -> cmake ../
make 
```

After installation, you can just use 'include "snn.h"' in your code, while compile it simply by linking libsnn.a, CBLAS and LAPACK library. 
For example, you can use g++ by 'g++ your_code.cpp libsnn.a -o output -llapacke -lgslcblas -lm -W'.


### User Guide

SNN has easy-to-use API, you can employ it on ``int``, ``float`` and ``double`` type data stored in column major order. The following is an example for loading the SNN and use the function. 

```c++
int rows = 10;
int cols = 3;
double df[rows*cols] = {
  0.5488135 , 0.54488318, 0.43758721, 0.38344152, 0.56804456,
  0.0871293 , 0.77815675, 0.79915856, 0.11827443, 0.94466892,
  0.71518937, 0.4236548 , 0.891773, 0.79172504, 0.92559664,
  0.0202184 , 0.87001215, 0.46147936, 0.63992102, 0.52184832,
  0.60276338, 0.64589411, 0.96366276, 0.52889492, 0.07103606,
  0.83261985, 0.97861834, 0.78052918, 0.14335329, 0.41466194
}; // data

 // index SNN model
SNN_MODEL<double, double> snn_model_Test(df, rows, cols);

// query data
double query[cols] = {0.5488135 , 0.71518937, 0.60276338}; 

// create the variable storing neighbors ID 
vector<int> knnID; 

// create the variable storing neighbors' distance to the query
vector<double> knnDist; 

 // employ query. the 0.4 refers to radius (range) 
snn_model_Test.radius_single_query(query, 0.4, &knnID, &knnDist);

