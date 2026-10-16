import numpy as np
class AgglomerativeClustering:
    def __init__(self, n_clusters: int, linkage : str = "complete"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.curr_clusters = None
        self.labels_ = np.array([])
        
    def fit(self, X: np.ndarray):
       self.curr_clusters = [[i] for i in range(len(X))]
       self.labels_ = np.empty(X.shape[0], dtype=int)
       
       while len(self.curr_clusters) > self.n_clusters:
           best_dist = float('inf')
           best_pair = (None, None)
           for j in range(len(self.curr_clusters)):
               for k in range((j+1), len(self.curr_clusters)):
                   dist = self.cluster_dist(X, self.curr_clusters[j], self.curr_clusters[k])
                   if dist < best_dist:
                       best_dist = dist
                       best_pair = (j,k)
           combined_clusters = []
           combined_clusters.extend(self.curr_clusters[best_pair[0]])
           combined_clusters.extend(self.curr_clusters[best_pair[1]])
           self.curr_clusters.pop(best_pair[-1])
           self.curr_clusters.pop(best_pair[0])
           self.curr_clusters.append(combined_clusters)
        
       for i, groups in enumerate(self.curr_clusters):
            self.labels_[groups] = i
            
   
    def cluster_dist(self, X: np.ndarray, cluster_a: np.ndarray, cluster_b: np.ndarray):
        difference_matrix = X[cluster_b][:, np.newaxis,:] - X[cluster_a][np.newaxis, :, :]
        squares = np.square(difference_matrix)
        add_squares = np.sum(squares, axis=-1)
        e_dist = np.sqrt(add_squares)
        return np.max(e_dist)
    
    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.labels_

if __name__ == "__main__":
    X = np.array([[0], [1], [5], [6], [12]])
    model = AgglomerativeClustering(2)
    print(model.fit_predict(X))
    