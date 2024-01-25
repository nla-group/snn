from .query import *
from .snn_cc import dfs1_cc, dfs2_cc, ndfs1_cc, ndfs2_cc, dfs3_cc
from ..sample import *
import numpy as np

class dbscan2:
    
    def __init__(self, algorithm='dbscan', eps=0.1, minPts=5, 
                    metric="euclidean", init='uniform',
                     sample_size=1000, memory_efficient=0, seed=42):
        
        self.algorithm = algorithm
        self.eps = eps
        self.minPts = minPts
        self.metric = metric
        self.init = init
        self.memory_efficient = memory_efficient
        self.sample_size = sample_size
        np.random.seed(seed)
        

    def fit_transform(self, X):
        self.fit(X)
        return self.labels
        
            
    def fit(self, X):
        
        snnm = build_snn_model(X)
        size = X.shape[0]
        
        if self.algorithm == 'dbscan':
            
            self.neighborhoods = snnm.radius_batch_query(X, 
                                       self.eps,
                                       return_distance=False)
            self.neighborhoods = list(self.neighborhoods.values())
            n_neighbors = np.asarray([len(neighbors) for neighbors in self.neighborhoods])
            corePoints = np.asarray(n_neighbors >= self.minPts, dtype=np.uint8)

            self.labels = np.full(size, -1, dtype=np.intp)
            ndfs1_cc(corePoints, self.neighborhoods, self.labels)
    
        elif self.algorithm == 'dbscan*':
            self.neighborhoods = snnm.radius_batch_query(X, 
                                       self.eps,
                                       return_distance=False)
            
            self.neighborhoods = list(self.neighborhoods.values())
            n_neighbors = np.asarray([len(neighbors) for neighbors in self.neighborhoods])
            corePoints = np.asarray(n_neighbors >= self.minPts, dtype=bool)
            self.labels = np.full(size, -1, dtype=np.intp)
            ndfs2_cc(corePoints, self.neighborhoods, self.labels)
            
        elif self.algorithm == 'dbscan++':
            if self.sample_size > size:
                self.sample_size = int(round(0.5*size))
                
            if self.init == 'uniform':
                subsampleID = uniform_sample(X, size=self.sample_size)
            else:
                subsampleID = greedy_k_center_sample(X, size=self.sample_size)
                
            subsampleID = uniform_sample(X, size=self.sample_size)
            if self.memory_efficient:
                self.neighborhoods = dict(snnm.radius_batch_query(X[subsampleID], self.eps, return_distance=False, memory_eff=1))
            else:
                self.neighborhoods = dict(snnm.radius_batch_query(X[subsampleID], self.eps, return_distance=False, memory_eff=0))
                
            n_neighbors = np.array([len(self.neighborhoods[neighbors]) for neighbors in self.neighborhoods])
            corePoints = np.zeros(size, dtype=np.uint8)
            corePoints[subsampleID[n_neighbors >= self.minPts]]=1
            self.labels = np.full(size, -1, dtype=np.intp)
            dfs3_cc(corePoints, self.neighborhoods, subsampleID, self.labels)
        
        
        
        
        
        
    
