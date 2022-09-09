## SNN: A lightweight fast exact radius query algorithm

[![C/C++ CI](https://github.com/nla-group/snn/actions/workflows/make.yml/badge.svg)](https://github.com/nla-group/snn/actions/workflows/make.yml)
[![!pypi](https://img.shields.io/pypi/v/snnpy?color=white)](https://pypi.org/project/snnpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SNN is a fast exact radius neareast neighbor search algorithm. It uses singular value decomposition to reduce the search space and speedup euclidean calculation with BLAS level 2 routine.  SNN enjoys faster speed with trivial python implementation compared to KDtree and Balltree in scikit-learn package. In this repository, we open source the C++ code of SNN with simple Cmake installing procedure. 



### Installation
SNN has dependencies on CBLAS, LAPACK and openMP, ensure install them before formally installing SNN. Reference LAPACK is [available from GitHub](https://github.com/Reference-LAPACK/lapack). LAPACK releases are also [available on netlib](http://www.netlib.org/lapack/).

Install SNN simply by, please modify the ``CMakeList.txt`` file according to your LAPACK location.  
```sh
git clone https://github.com/nla-group/snn.git
cd snn
cmake . # or mkdir build -> cd build -> cmake ../
make 
cp *.a /usr/lib
cp include/*.h /usr/include
```

After installation, you can just use ``include "snn.h"`` in your code, while compile it simply by linking libsnn.a, CBLAS and LAPACK library. 
For example, you can use g++ by ``g++ your_code.cpp libsnn.a -o output -llapacke -lgslcblas -lm -W`` in Ubuntu.


### User Guide

SNN has easy-to-use API, you can employ it on ``int``, ``float`` and ``double`` type data stored in column major order. The following is an example for loading the SNN and use the function. 

We fist prepare the data:
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
}; 

/* data -- column major order

0.548813 0.715189 0.602763 
0.544883 0.423655 0.645894 
0.437587 0.891773 0.963663 
0.383442 0.791725 0.528895 
0.568045 0.925597 0.0710361 
0.0871293 0.0202184 0.83262 
0.778157 0.870012 0.978618 
0.799159 0.461479 0.780529 
0.118274 0.639921 0.143353 
0.944669 0.521848 0.414662 
*
```

Then, we index the SNN model, specify the number of objects and feature dimensions in the data:
```c++
// index SNN model
SNN_MODEL<double, double> snn_model_Test(df, rows, cols);
```


Query simple query as below:
```c++
// query data
double query[cols] = {0.5488135 , 0.71518937, 0.60276338}; 

// create the variable storing neighbors ID 
vector<int> knnID; 

// create the variable storing neighbors' distance to the query
vector<double> knnDist; 

// employ single query, the 0.4 refers to radius (range) 
snn_model_Test.radius_single_query(query, 0.4, &knnID, &knnDist);
```


We can also query multiple objects:
```c++
 // employ two queries, parallel compute by openMP
double query_batch[2*cols] = {0.5488135, 0.944669, 0.71518937, 0.521848, 0.60276338, 0.414662};

vector<vector<int> > batch_knnID;
vector<vector<double> > batch_knnDist;

snn_model_Test.radius_batch_query(query_batch, 0.4, &batch_knnID, &batch_knnDist, 2);

for (int j=0; j<2; j++){
    cout << "knnID" << endl;
    for (auto i: batch_knnID[j]){
        cout << i << " ";
    }

    cout << "\nknnDist" << endl;
    for (auto i: batch_knnDist[j]){
        cout << i << " ";
    }

    cout << endl;
}

/* output
  knnID
  3 0 1 7 
  knnDist
  0.196627 0 0.294734 0.398299 

  knnID
  9 7 
  knnDist
  3.35193e-07 0.398342
*/
```


### Python API

You can also use python trivial implementation of SNN, simply install 

```sh
pip install snnpy
```

The example illustrates the use of SNN:

```python
import numpy as np
from snnpy import *

n_samples = 500000
n_dim =  100
radius = 3.8
rng = np.random.RandomState(0)
X = rng.random_sample((n_samples, n_dim))  

# index SNN model
snn_model = build_snn_model(X)   

# query data
ind, dist = query_radius(X[0], snn_model, radius)

sort_id = np.argsort(dist)

# return top 5
print("ID:", ", ".join([str(i) for i in ind[sort_id][:5]]))

# return top 10
print("distance:", ", ".join([str(i) for i in dist[sort_id][:5]]))

```



### License
All the content in this repository is licensed under the MIT License.

Copyright © 2022 NLA group

