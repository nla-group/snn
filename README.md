## SNN: Fast and exact fixed-radius neighbor search

[![C/C++ CI](https://github.com/nla-group/snn/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/nla-group/snn/actions/workflows/c-cpp.yml)
[![!pypi](https://img.shields.io/pypi/v/snnpy?color=white)](https://pypi.org/project/snnpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/532659733.svg)](https://zenodo.org/doi/10.5281/zenodo.10275013)


SNN is a fast and exact fixed-radius nearest neighbor search algorithm [1]. It uses the first principal component of the data to prune the search space and speeds up Euclidean distance computations using high-level BLAS routines. SNN is implemented in native Python. On many problems, SNN is faster than KDtree and Balltree in the scikit-learn package. There is also a C++ implementation of SNN. 

### Reproducibility

To reproduce the experiments from the paper [1], see the instructions and code in the `exp` subfolder.

### Python installation

The native Python implementation of SNN can be installed by:

```sh
pip install snnpy
```

### Python API

Here is an example illustrating the use of SNN:

```python
import numpy as np
from snnpy import *
from time import time

n_samples = 100000
n_dim =  100
radius = 3.5
rng = np.random.RandomState(0)
X = rng.random_sample((n_samples, n_dim))  

# build SNN model
st = time()
snn_model = build_snn_model(X)  
print("SNN index time:", time()-st)
# will be faster if return_dist is False, then no distance information come out

# query neighbors of X[0]
st = time()
ind,dist = snn_model.query_radius(X[0], radius, return_distance=True)
# If remove the returning of the associated distance, use: ind, dist = snn_model.query_radius(X[0], radius, return_distance=False)
sort_ind = np.argsort(dist)
print("SNN query time:", time()-st)

# print total number and top five indices
print("number of neighbors:", len(ind))
print("indices of closest five:", ", ".join([str(i) for i in ind[sort_ind][:5]]))

# EXAMPLE OUTPUT
# SNN index time: 0.2224433422088623
# SNN query time: 0.009207725524902344
# number of neighbors: 550
# indices of closest five: 0, 27279, 69983, 65906, 97095
```

Compare this to sklearn's KDTree:

```python
from sklearn.neighbors import KDTree
st = time()
tree = KDTree(X)    
print("KDTree index time:", time()-st)
st = time()
ind2 = tree.query_radius(X[0].reshape(1, -1), radius)
print("KDTree query time:", time()-st)
print("number of neighbors:", len(ind2[0]))

# KDTree index time: 7.597502946853638
# KDTree query time: 0.08962678909301758
# number of neighbors: 550
```


We also support multi-point queries which exploit single-threaded BLAS-3 (multi-threading is under development):

```python
ind = snn_model.radius_batch_query(X[:10], radius) 
```

``snn_model.radius_batch_query`` uses Numba for further speedup. For the first run, Numba has to go through the code and optimize it, adding extra overhead. Each subsequent run of batch queries will be much faster.

### C++ installation

The C++ version of SNN has dependencies on CBLAS, LAPACK and openMP. Reference LAPACK is [available from GitHub](https://github.com/Reference-LAPACK/lapack). LAPACK releases are [available on netlib](http://www.netlib.org/lapack/) (using [MKL](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html) can yield further speedup).

Please modify the ``CMakeList.txt`` file to reflect your LAPACK location:
```sh
git clone https://github.com/nla-group/snn.git
cd snn
cmake . # or mkdir build -> cd build -> cmake ../
make 
cp *.a /usr/lib
cp include/*.h /usr/include
```

After installation, you can ``include "snn.h"`` in your code, and compile it by linking the libsnn.a, CBLAS and LAPACK libraries. 
For example, you can use g++ with GSL BLAS and LAPACK by typing ``g++ your_code.cpp libsnn.a -o output -llapacke -lgslcblas -lm -W`` in Ubuntu.

### C++ API

SNN has an easy-to-use API, and it works with ``int``, ``float`` and ``double`` data types stored in column major order. 

We first prepare the data:
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

Now we create the SNN index, specifying the number of objects and feature dimensions in the data:
```c++
// index SNN model
SNN_MODEL<double, double> snn_model_test(df, rows, cols);
```

Querying neighbors of a single data point:
```c++
// query data
double query[cols] = {0.5488135 , 0.71518937, 0.60276338}; 

// create the variable storing neighbors ID 
vector<int> knnID; 

// create the variable storing neighbors' distance to the query
vector<double> knnDist; 

// employ single query, the number 0.4 refers to the search radius (range) 
snn_model_test.radius_single_query(query, 0.4, &knnID, &knnDist);
```

We can also query neighbors of multiple data points at once:
```c++
 // employ two queries, parallel compute by openMP
double query_batch[2*cols] = {0.5488135, 0.944669, 0.71518937, 0.521848, 0.60276338, 0.414662};

vector<vector<int> > batch_knnID;
vector<vector<double> > batch_knnDist;

snn_model_test.radius_batch_query(query_batch, 0.4, &batch_knnID, &batch_knnDist, 2);

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

### License
All the content in this repository is licensed under the MIT License. 


## Reference
```bibtex
@article{CG24b,
  title   = {Fast and exact fixed-radius neighbor search based on sorting},
  author  = {Chen, Xinye and G\"{u}ttel, Stefan},
  year    = {2024},
  volume  = {},
  number  = {},
  pages   = {},
  journal = {To appear in PeerJ Computer Science},
  url     = {https://arxiv.org/abs/2212.07679},
  webpdf  = {https://arxiv.org/abs/2212.07679}
}
```

