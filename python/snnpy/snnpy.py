# MIT License
# Copyright (c) 2022 Stefan Güttel, Xinye Chen
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import numpy as np
from scipy.linalg import get_blas_funcs, eigh
    

def euclid(xxt, X, v):
    return (xxt + np.inner(v,v).ravel() -2*X.dot(v)).astype(float)


def euclid_batch(xxt, inner_queries, ddata_queries, i):
    return (xxt + inner_queries[i] - 2*ddata_queries[i])


def euclid_batch_mf(xxt, inner_queries, ddata_query, i):
    return (xxt + inner_queries[i] - 2*ddata_query)


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
    

    def query_radius((self, query, radius, return_distance=False):
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
            return self._radius_batch_query_mf(queries, radius, return_distance)
        
        else:
            return self._radius_batch_query(queries, radius, return_distance)
        
        
    def _radius_batch_query(self, queries, radius, return_distance=False):
        queries = np.subtract(queries, self.mu)
        sv_qs = np.inner(queries, self.v)
        lefts = np.searchsorted(self.sort_vals, sv_qs-radius)
        rights = np.searchsorted(self.sort_vals, sv_qs+radius)

        inner_queries = np.einsum('ij,ij->i', queries, queries) 
        # extend from np.inner(v,v).ravel(), 1D array

        ddata_queries = np.inner(queries, self.data)
        # extend from self.data.dot(queries[0]), 2D array (n_samples, n_samples)

        knn_ind = dict()
        
        num = queries.shape[0]
        radius = radius**2
        
        if return_distance:
            knn_dist = dict()
            
            for i in range(num):
                batch_dist_set = euclid_batch(self.xxt,
                                              inner_queries, 
                                              ddata_queries,
                                              i
                                             )[lefts[i]:rights[i]]

                filter_radius = batch_dist_set <= radius
                knn_ind[i] = self.sort_id[lefts[i]:rights[i]][filter_radius]
                
                
                knn_dist[i] = np.sqrt(batch_dist_set[filter_radius])
                
                
            return knn_ind, knn_dist

        else:
            for i in range(num):
                batch_dist_set = euclid_batch(self.xxt,
                                              inner_queries, 
                                              ddata_queries,
                                              i
                                             )[lefts[i]:rights[i]]

                knn_ind[i] = self.sort_id[lefts[i]:rights[i]][batch_dist_set <= radius]

            return knn_ind
        
        
    def _radius_batch_query_mf(self, queries, radius, return_distance=False): # memory efficient
        queries = np.subtract(queries, self.mu)
        sv_qs = np.inner(queries, self.v)
        lefts = np.searchsorted(self.sort_vals, sv_qs-radius)
        rights = np.searchsorted(self.sort_vals, sv_qs+radius)

        inner_queries = np.einsum('ij,ij->i', queries, queries) 
        # extend from np.inner(v,v).ravel(), 1D array

        # ddata_queries = np.inner(queries, self.data)
        # extend from self.data.dot(queries[0]), 2D array (n_samples, n_samples)

        knn_ind = dict()
        
        num = queries.shape[0]
        radius = radius**2
        
        if return_distance:
            knn_dist = dict()
            
            for i in range(num):
                ddata_query = self.data.dot(queries[i]) 
                batch_dist_set = euclid_batch_mf(self.xxt,
                                              inner_queries, 
                                              ddata_query,
                                              i
                                             )[lefts[i]:rights[i]]

                filter_radius = batch_dist_set <= radius
                knn_ind[i] = self.sort_id[lefts[i]:rights[i]][filter_radius]
                
                
                knn_dist[i] = np.sqrt(batch_dist_set[filter_radius])
                
            return knn_ind, knn_dist

        else:
            for i in range(num):
                ddata_query = self.data.dot(queries[i]) 
                batch_dist_set = euclid_batch_mf(self.xxt,
                                              inner_queries, 
                                              ddata_query,
                                              i
                                             )[lefts[i]:rights[i]]

                knn_ind[i] = self.sort_id[lefts[i]:rights[i]][batch_dist_set <= radius]

            return knn_ind
        
