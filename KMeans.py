import numpy as np
class KMeans:
    def __init__(self, clusters):
        self.clusters = clusters
        self.curr_centroids = None
    
    def fit(self, X: np.ndarray):
        rng = np.random.default_rng()
        self.curr_centroids = rng.choice(X, size=self.clusters, replace=False)
        while True:
            copy = self.curr_centroids.copy()
            difference_matrix = self.curr_centroids[:, np.newaxis,:] - X[np.newaxis, :, :]
            squares = np.square(difference_matrix)
            add_squares = np.sum(squares, axis=-1)
            e_dist = np.sqrt(add_squares)
            min_idx = np.argmin(e_dist, axis=0)
            
            for i in range(self.clusters):
                idx = min_idx == i
                cluster_pts = X[idx]
                mean = np.mean(cluster_pts, axis=0)
                self.curr_centroids[i] = mean
                
            if np.allclose(copy, self.curr_centroids):
                break  
    
    def predict(self, X:np.ndarray):
        difference_matrix = self.curr_centroids[:, np.newaxis,:] - X[np.newaxis, :, :]
        squares = np.square(difference_matrix)
        add_squares = np.sum(squares, axis=-1)
        e_dist = np.sqrt(add_squares)
        min_idx = np.argmin(e_dist, axis=0)
        return min_idx
        
        
if __name__ == "__main__":
    model = KMeans(2)
    X = np.array([
        [1.0, 1.0],
        [1.5, 2.0],
        [2.0, 1.5],
        [8.0, 8.0],
        [8.5, 9.0],
        [9.0, 8.5],])
    print(X)
    print(model.predict(X))
