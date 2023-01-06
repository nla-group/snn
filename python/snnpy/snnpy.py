# MIT License
# Copyright (c) 2022 Stefan Güttel, Xinye Chen
# See license file for details



import numba
import numpy as np
from scipy.linalg import get_blas_funcs, eigh
    

def euclid(xxt, X, v):
    return (xxt + np.inner(v,v).ravel() -2*X.dot(v)).astype(float)



# template for memory efficient computation - deprecated
def euclid_batch(xxt, inner_queries, ddata_queries, i):
    return (xxt + inner_queries[i] - 2*ddata_queries[i])


# template for memory efficient computation - deprecated
def euclid_batch_mf(xxt, inner_queries, ddata_query, i):
    return (xxt + inner_queries[i] - 2*ddata_query)


@numba.njit(cache=False)
def bisection_sort(queries, mu, v, sort_vals, radius):
    queries = np.subtract(queries, mu)
    sv_qs = np.dot(queries, v)
    lefts = np.searchsorted(sort_vals, sv_qs-radius)
    rights = np.searchsorted(sort_vals, sv_qs+radius)
    return queries, lefts, rights


@numba.njit(cache=False)
def query_batches1(queries, data, inner_queries, xxt, radius, lefts, rights, sort_id):
    knn_ind = dict()
    ddata_queries = np.dot(queries, data.T)
    for i in range(queries.shape[0]):
        batch_dist_set = (xxt + inner_queries[i] - 2*ddata_queries[i])

        batch_dist_set = batch_dist_set[lefts[i]:rights[i]]
        knn_ind[i] = sort_id[lefts[i]:rights[i]][batch_dist_set <= radius]
    
    return knn_ind


@numba.njit(cache=False)
def query_batches2(queries, data, inner_queries, xxt, radius, lefts, rights, sort_id):
    knn_ind = dict()
    knn_dist = dict()
    ddata_queries = np.dot(queries, data.T)
    for i in range(queries.shape[0]):
        batch_dist_set = (xxt + inner_queries[i] - 2*ddata_queries[i])

        batch_dist_set = batch_dist_set[lefts[i]:rights[i]]

        filter_radius = batch_dist_set <= radius
        knn_ind[i] = sort_id[lefts[i]:rights[i]][filter_radius]
        knn_dist[i] = np.sqrt(batch_dist_set[filter_radius])
        
    return knn_ind, knn_dist


@numba.njit(cache=False)
def query_batches1_mef(queries, data, inner_queries, xxt, radius, lefts, rights, sort_id):
    knn_ind = dict()
    
    for i in range(queries.shape[0]):
        ddata_query = np.dot(data, queries[i])
        batch_dist_set = (xxt + inner_queries[i] - 2*ddata_query)
        batch_dist_set = batch_dist_set[lefts[i]:rights[i]]
        
        knn_ind[i] = sort_id[lefts[i]:rights[i]][batch_dist_set <= radius]
    
    return knn_ind


@numba.njit(cache=False)
def query_batches2_mef(queries, data, inner_queries, xxt, radius, lefts, rights, sort_id):
    knn_ind = dict()
    knn_dist = dict()
    
    for i in range(queries.shape[0]):
        ddata_query = np.dot(data, queries[i]) 
        batch_dist_set = (xxt + inner_queries[i] - 2*ddata_query)
        batch_dist_set = batch_dist_set[lefts[i]:rights[i]]

        filter_radius = batch_dist_set <= radius
        knn_ind[i] = sort_id[lefts[i]:rights[i]][filter_radius]
        knn_dist[i] = np.sqrt(batch_dist_set[filter_radius])
        
    return knn_ind, knn_dist

    
def _radius_batch_query(mu, v, xxt, sort_vals, sort_id, data, queries, radius, return_distance=False):
    queries, lefts, rights = bisection_sort(queries, mu, v, sort_vals, radius)
    inner_queries = np.einsum('ij,ij->i', queries, queries) 
    knn_ind = dict()
    radius = radius**2
    
    if return_distance:
        return query_batches2(queries, data, inner_queries, xxt, radius, lefts, rights, sort_id)

    else:
        return query_batches1(queries, data, inner_queries, xxt, radius, lefts, rights, sort_id)
    

def _radius_batch_query_mef(mu, v, xxt, sort_vals, sort_id, data, queries, radius, return_distance=False):
    queries, lefts, rights = bisection_sort(queries, mu, v, sort_vals, radius)
    inner_queries = np.einsum('ij,ij->i', queries, queries) 
    knn_ind = dict()
    radius = radius**2
    
    if return_distance:
        return query_batches2_mef(queries, data, inner_queries, xxt, radius, lefts, rights, sort_id)

    else:
        return query_batches1_mef(queries, data, inner_queries, xxt, radius, lefts, rights, sort_id)
    
    
    

class build_snn_model:
    def __init__( self, data, n_jobs=1, verbose=1):
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.mu = data.mean(axis=0)
        data = data - self.mu
        if data.shape[1]>1 and data.shape[1]<=2 :
            gemm = get_blas_funcs("gemm", [data.T, data])
            dTd = gemm(1, data.T, data)
            _, v = eigh(dTd, subset_by_index=[data.shape[1]-1, data.shape[1]-1])
            sort_vals = data@v.reshape(-1)
            
        elif data.shape[1]>2:
            dTd = np.dot(data.T, data)
            _, v = eigh(dTd, subset_by_index=[data.shape[1]-1, data.shape[1]-1])
            sort_vals = data@v.reshape(-1)
            
        else:
            sort_vals = data[:,0].reshape(-1)
        
        self.sort_id = np.argsort(sort_vals)
        self.sort_vals = sort_vals[self.sort_id]
        self.data = data[self.sort_id]
        self.v = v.reshape(-1)
        self.xxt = np.einsum('ij,ij->i', self.data, self.data) # np.linalg.norm(X, axis=1)**2
    

    def query_radius(self, query, radius, return_distance=False):
        query = np.subtract(query, self.mu)
        sv_q = np.inner(query, self.v)
        left = np.searchsorted(self.sort_vals, sv_q-radius)
        right = np.searchsorted(self.sort_vals, sv_q+radius)
        dist_set =  euclid(self.xxt[left:right], self.data[left:right], query)

        filter_radius = dist_set <= radius**2
        knn_ind = self.sort_id[left:right][filter_radius]

        if return_distance:
            knn_dist = np.sqrt(dist_set[filter_radius])
            return knn_ind, knn_dist
        else:
            return knn_ind
        
        
    def radius_batch_query(self, queries, radius, return_distance=False, memory_eff=0):
        if memory_eff:
            return _radius_batch_query_mef(self.mu, 
                                       self.v, 
                                       self.xxt, 
                                       self.sort_vals, 
                                       self.sort_id, self.data, 
                                       queries, radius, return_distance)
            
        else:
            return _radius_batch_query(self.mu, 
                                       self.v, 
                                       self.xxt, 
                                       self.sort_vals, 
                                       self.sort_id, self.data, 
                                       queries, radius, return_distance)
        
