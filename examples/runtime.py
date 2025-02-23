import numpy as np
from snnpy import *
from time import time

import warnings
warnings.filterwarnings("ignore")


n_samples = 100000
n_dim =  100
radius = 2
rng = np.random.RandomState(0)
X = rng.random_sample((n_samples, n_dim))  

# build SNN model
st = time()
snn_model = build_snn_model(X)  
print("SNN index time:", time()-st)
# will be faster if return_dist is False, then no distance information come out

# query neighbors of X[0]
print("\nTesting single query (float64):")
st = time()
for i in range(1000):
    ind, dist = snn_model.query_radius(X[1050+i], radius, return_distance=True)
# If remove the returning of the associated distance, use: ind, dist = snn_model.query_radius(X[0], radius, return_distance=False)
# sort_ind = np.argsort(dist)
print("SNN query time:", time()-st)
print(ind)


# print total number and top five indices
#print("number of neighbors:", len(ind))
#print("indices of closest five:", ", ".join([str(i) for i in ind[sort_ind][:5]]))

# EXAMPLE OUTPUT
# SNN index time: 0.2224433422088623
# SNN query time: 0.009207725524902344
# number of neighbors: 550
# indices of closest five: 0, 27279, 69983, 65906, 97095


# test new
print("\nTesting batch query (float64):")
st = time()
snn_model = build_snn_model(X)
print("SNN index time:", time()-st)
st = time()
results = snn_model.batch_query_radius(X[1050:(1050+1000)], radius)
print("SNN query time:", time()-st)
print(results[-1])



# test openmp (beta)
print("\nTesting SNN_FLOAT (float32) - beta:")
snn_omp = snnomp.SNN_FLOAT(X.astype(float), num_threads=40)
results = snn_omp.query_radius_batch(X[1050:(1050+1000)], radius)
print(results[-1])
print("SNN query time:", time()-st)

print("\nTesting SNN_DOUBLE (float64) - beta:")
snn_omp = snnomp.SNN_DOUBLE(X, num_threads=40)
results = snn_omp.query_radius_batch(X[1050:(1050+1000)], radius)
print("SNN query time:", time()-st)
print(results[-1])
