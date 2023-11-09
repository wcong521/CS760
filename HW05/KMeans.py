import numpy as np
import math

class KMeans:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X):

        # initialize centroids
        centroid_indicies = np.random.choice(X.shape[0], self.k, replace=False)
        centroids = X[centroid_indicies]

        delta = math.inf
        prev_loss = math.inf

        while delta > .001:

            # cluster assignment
            clusters = []
            for x in X:
                cluster_index = np.argmin([np.linalg.norm(x - c) for c in centroids])
                clusters.append(cluster_index)

            # centroid refitting
            for i in range(self.k):
                cluster_indicies = np.where(np.array(clusters) == i)
                centroids[i] = np.mean(X[cluster_indicies], axis=0)
            
            # compute loss 
            loss = 0
            for i in range(len(X)):
                loss += np.linalg.norm(X[i] - centroids[clusters[i]]) ** 2

            delta = prev_loss - loss
            prev_loss = loss

        return clusters, centroids, loss




        
