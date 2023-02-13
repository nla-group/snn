import numpy as np
from numpy import linalg


def calculate_shortest_distance(data, centers):
    distance = np.empty([data.shape[0], centers.shape[0]])
    for i in range(centers.shape[0]):
        distance[:, i] = linalg.norm(data - centers[i], axis=1)
        
    return np.min(distance, axis=1)


def uniform_sample(X, size=100):
    """
    initialized the centroids with uniform initialization
    
    inputs:
        X - numpy array of data points having shape (n_samples, n_dim)
        size - number of clusters
    """
    
    subsampleID = np.random.choice(X.shape[0], size=size, replace=False)
    return subsampleID
    
    
def greedy_k_center_sample(X, size=100):
    '''
    initialized the centroids with greedy k center initialization
    
    inputs:
        X - numpy array of data points having shape (n_samples, n_dim)
        size - number of clusters
    '''

    subsampleID = np.empty(size, dtype=int)
    subsampleID[0] = np.random.randint(X.shape[0])
    for c_id in range(1, size):
        shortest_distance = calculate_shortest_distance(X, X[subsampleID[:c_id]])
        subsampleID[c_id] = np.argmax(shortest_distance)
        
    return np.array(subsampleID)