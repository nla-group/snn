import numpy as np

def brute_force_search_k(query, X, k):
    dist = np.linalg.norm(X - query, axis=1)
    candidates = np.argsort(dist)[:k]
    distances = dist[candidates]
    return candidates, distances
    
    
    
def brute_force_search_radius(query, X, radius):
    dist = np.linalg.norm(X - query, axis=1)
    
    filter_radius = dist <= radius
    distances = dist[filter_radius] 
    return np.where(filter_radius)[0], distances



def bf_radius_fairness(query, data, radius, return_distance=False):
    dist_set =  euclid(data, query)
    filter_radius = dist_set <= radius**2
    knn_ind = np.where(filter_radius)[0]
    if return_distance:
        return knn_ind, np.sqrt(dist_set[filter_radius])
    else:
        return knn_ind

def euclid(X, v):
    return np.einsum('ij,ij->i',X,X) + np.inner(v,v).ravel() -2*X.dot(v)
