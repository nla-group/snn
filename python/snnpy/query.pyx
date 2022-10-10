
### unused, to be developed

import numpy as np
cimport numpy as np


cdef euclid(np.ndarray[np.double_t, ndim=1] xxt,
            np.ndarray[np.double_t, ndim=2] X, 
            np.ndarray[np.double_t, ndim=1] v):
    
    return (xxt + np.inner(v,v).ravel() -2*X.dot(v)).astype(float)



cdef euclid_batch(np.ndarray[np.double_t, ndim=1] xxt, 
                  np.ndarray[np.double_t, ndim=1] inner_queries, 
                  np.ndarray[np.double_t, ndim=2] ddata_queries, 
                  int i):
    return (xxt + inner_queries[i] - 2*ddata_queries[i])



cpdef _cy_batch_query_radius( np.ndarray[np.double_t, ndim=2] data, 
                             np.ndarray[np.double_t, ndim=2] queries, 
                             np.ndarray[np.double_t, ndim=1] mu,
                             np.ndarray[np.npy_intp, ndim=1] sort_id, 
                             np.ndarray[np.double_t, ndim=1] sort_vals, 
                             np.ndarray[np.double_t, ndim=1] v,
                             np.ndarray[np.double_t, ndim=1] xxt,
                             double radius, 
                             np.intp_t return_dist):
    
    cdef list knn_ind = list()
    cdef list knn_dist
    
    queries = np.subtract(queries, mu)
    cdef np.ndarray[np.double_t, ndim=1] sv_qs = np.inner(queries, v)
    
    cdef np.ndarray[np.npy_intp, ndim=1] lefts = np.searchsorted(sort_vals, sv_qs-radius)
    cdef np.ndarray[np.npy_intp, ndim=1] rights = np.searchsorted(sort_vals, sv_qs+radius)

    cdef np.ndarray[np.double_t, ndim=1] inner_queries = np.einsum('ij,ij->i', queries, queries) 

    cdef np.ndarray[np.double_t, ndim=2] ddata_queries = np.inner(queries, data)
    cdef np.ndarray[np.npy_bool, ndim=1, cast=True] filter_radius
    cdef np.intp_t i
    cdef np.intp_t num = queries.shape[0]
    radius = radius**2
    
    if return_dist:
        knn_dist = list()
        for i in range(num):
            batch_dist_set = euclid_batch(xxt,
                                          inner_queries, 
                                          ddata_queries,
                                          i
                                         )[lefts[i]:rights[i]]

            filter_radius = batch_dist_set <= radius
            knn_ind.append(
                sort_id[lefts[i]:rights[i]][filter_radius]
            )

            knn_dist.append(
                np.sqrt(batch_dist_set[filter_radius])
            )

        return knn_ind, knn_dist

    else:
        for i in range(num):
            batch_dist_set = euclid_batch(xxt,
                                          inner_queries, 
                                          ddata_queries,
                                          i
                                         )[lefts[i]:rights[i]]

            filter_radius = batch_dist_set <= radius
            knn_ind.append(
                sort_id[lefts[i]:rights[i]][filter_radius]
            )

        return knn_ind
