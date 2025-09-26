## SNN: Fast and exact fixed-radius neighbor search

[![!pypi](https://img.shields.io/pypi/v/snnpy?color=white)](https://pypi.org/project/snnpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/532659733.svg)](https://zenodo.org/doi/10.5281/zenodo.10275013)


SNN is a fast and exact fixed-radius nearest neighbor search algorithm [1]. It uses the first principal component of the data to prune the search space and speeds up Euclidean distance computations using high-level BLAS routines. SNN is implemented in native Python. On many problems, SNN is faster than KDtree and Balltree in the scikit-learn package. 

### Reproducibility

To reproduce the experiments from the paper [1], see the instructions and code in the `exp` subfolder.

### Python installation

The native Python implementation of SNN can be installed by:

```sh
pip install snnpy
```

### Usage

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

### License
All the content in this repository is licensed under the MIT License. 


## Reference

```
Chen X, Güttel S. 2024. Fast and exact fixed-radius neighbor search based on sorting. PeerJ Computer Science 10:e1929 https://doi.org/10.7717/peerj-cs.1929
```

