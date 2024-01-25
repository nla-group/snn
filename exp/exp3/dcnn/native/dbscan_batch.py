from sklearn.neighbors import NearestNeighbors
from .native_cc import dfs1_cc, dfs2_cc, dfs3_cc
from ..sample import *
import numpy as np

# for exp

class dbscan:
    
    def __init__(self, algorithm='dbscan', eps=0.1, minPts=5,
                    metric="euclidean", query="kd_tree", sample_size=1000,
                 leaf_size=30, n_jobs=1, seed=42):
        
        self.algorithm = algorithm
        self.eps = eps
        self.minPts = minPts
        self.metric = metric
        self.query = query
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.sample_size = sample_size
        np.random.seed(seed)

    def fit_transform(self, X):
        self.fit(X)
        return self.labels
        
            
    def fit(self, X):
        size = X.shape[0]
        neighborsModel = NearestNeighbors(
            radius=self.eps,
            algorithm=self.query,
            leaf_size=self.leaf_size,
            metric=self.metric,
            n_jobs=self.n_jobs
        )

        neighborsModel.fit(X)
        
        if self.algorithm == 'dbscan':
            
            # if self.query == 'kd_tree':
                # neighborsModel = KDTree(X, leaf_size=self.leaf_size)
            #     neighborhoods=[neighborsModel.radius_neighbors(X[i:i+1], self.eps, 
            #                                 return_distance=False)[0] for i in range(size)]
            # elif self.query == 'ball_tree':
            #     # neighborsModel = BallTree(X, leaf_size=self.leaf_size)
            #     neighborhoods=[neighborsModel.radius_neighbors(X[i:i+1], self.eps, 
            #                                 return_distance=False)[0] for i in range(size)]
            # else:

            self.neighborhoods = neighborsModel.radius_neighbors(X, radius=self.eps, 
                          return_distance=False).tolist()

            n_neighbors = np.array([len(neighbors) for neighbors in self.neighborhoods])
            corePoints = np.asarray(n_neighbors >= self.minPts, dtype=np.uint8)
            self.labels = np.full(size, -1, dtype=np.intp)
            dfs1_cc(corePoints, self.neighborhoods, self.labels)
    
        elif self.algorithm == 'dbscan*':

            self.neighborhoods=  neighborsModel.radius_neighbors(X, self.eps, 
                                            return_distance=False).tolist()
            
            n_neighbors = np.array([len(neighbors) for neighbors in self.neighborhoods])
            corePoints = np.asarray(n_neighbors >= self.minPts, dtype=bool)
            self.labels = np.full(size, -1, dtype=np.intp)
            dfs2_cc(corePoints, self.neighborhoods, self.labels)
            
        elif self.algorithm == 'dbscan++':
            if self.sample_size > size:
                self.sample_size = int(round(0.5*size))
                
            subsampleID = uniform_sample(X, size=self.sample_size)
            self.neighborhoods = neighborsModel.radius_neighbors(X[subsampleID], self.eps,
                                                            return_distance=False).tolist()
            n_neighbors = np.array([len(neighbors) for neighbors in self.neighborhoods])
            corePoints = np.zeros(size, dtype=np.uint8)
            corePoints[subsampleID[n_neighbors >= self.minPts]]=1
            self.labels = np.full(size, -1, dtype=np.intp)
            dfs3_cc(corePoints, self.neighborhoods, subsampleID, self.labels)
    
        
        
        
        
        
        
    
