# import numba
import numpy as np
from scipy.linalg import get_blas_funcs, eigh




class build_snn_model:
    def __init__( self, data, verbose=1):
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
    
    
    def query_radius(self, query, r, return_distance=False):
        query = np.subtract(query, self.mu)
        sv_q = np.inner(query, self.v)
        left = np.searchsorted(self.sort_vals, sv_q-r)
        right = np.searchsorted(self.sort_vals, sv_q+r)
        dist_set =  euclid(self.xxt[left:right], self.data[left:right], query)

        filter_r = dist_set <= r**2
        knn_ind = self.sort_id[left:right][filter_r]

        if return_distance:
            knn_dist = np.sqrt(dist_set[filter_r])
            return knn_ind, knn_dist
        else:
            return knn_ind
        
    
    def batch_query_radius1(self, queries, r, return_distance=False):
        """
        Efficiently handles multiple queries at once.
        - queries: (m, d) array of query points
        - r: radius for nearest neighbors
        - return_distance: if True, also returns distances
        """
        queries = queries - self.mu  # Center queries
        sv_q = queries @ self.v  # (m,) projections of queries onto PC
        left = np.searchsorted(self.sort_vals, sv_q - r, side='left')
        right = np.searchsorted(self.sort_vals, sv_q + r, side='right')

        # Compute squared norms of queries
        queries_xxt = np.sum(queries**2, axis=1)

        results = []
        distances = []

        r = r**2
        for i in range(queries.shape[0]):  # Iterate over multiple queries
            idx_range = slice(left[i], right[i])
            data_subset = self.data[idx_range]
            xxt_subset = self.xxt[idx_range]

            # Compute squared distances efficiently
            dists = xxt_subset + queries_xxt[i] - 2 * np.dot(data_subset, queries[i])
            mask = dists <= r  # Radius filter
            knn_idx = self.sort_id[idx_range][mask]
            
            results.append(knn_idx)
            if return_distance:
                distances.append(np.sqrt(dists[mask]))

        return (results, distances) if return_distance else results
    

    def batch_query_radius2(self, queries, r, return_distance=False):
        queries = queries - self.mu
        sv_q = queries @ self.v  # Project all queries onto principal component
        r_sq = r ** 2

        # Use searchsorted in a vectorized manner
        left = np.searchsorted(self.sort_vals, sv_q - r)
        right = np.searchsorted(self.sort_vals, sv_q + r)

        results = []
        distances = []

        for i in range(len(queries)):
            idx_range = slice(left[i], right[i])
            dist_sq = self.xxt[idx_range] + np.sum(queries[i] ** 2) - 2 * (self.data[idx_range] @ queries[i])

            mask = dist_sq <= r_sq
            indices = self.sort_id[idx_range][mask]

            if return_distance:
                results.append(indices)
                distances.append(np.sqrt(dist_sq[mask]))
            else:
                results.append(indices)

        return (results, distances) if return_distance else results
    

def euclid(xxt, X, v):
    return (xxt + np.inner(v,v).ravel() -2*X.dot(v)).astype(float)

