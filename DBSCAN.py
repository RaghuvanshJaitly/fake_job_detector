import numpy as np

class DBSCAN:

    def __init__(self, eps: float, min_samples=5):
        #distance between points/neighbours
        self.eps = eps
        self.min_samples = min_samples
        self.cluster_ids = None
    
    def fit(self, X: np.ndarray):
        self.labels = np.full(X.shape[0], -2)
        self.visited = np.full(X.shape[0], False)
        self.cluster_ids = 0

        for i in range(len(X)):
            if not(self.visited[i]):
                self.visited[i] = True
                satisfy_eps_idx = self.calc_edist_neighbour(X, i)
                if len(satisfy_eps_idx) < self.min_samples:
                    self.labels[i] = -1
                else:
                    self.labels[i] = self.cluster_ids
                    worklist = satisfy_eps_idx.tolist()
                    position_counter = 0
                    while position_counter < len(worklist):
                        curr_idx = worklist[position_counter]
                        if not(self.visited[curr_idx]):
                            self.visited[curr_idx] = True
                            neighbours = self.calc_edist_neighbour(X, curr_idx)
                            if len(neighbours) >= self.min_samples:
                                for j in range(len(neighbours)):
                                    if neighbours[j] not in worklist:
                                        worklist.append(neighbours[j])
                        if self.labels[curr_idx] == -1 or self.labels[curr_idx] == -2:
                            self.labels[curr_idx] = self.cluster_ids
                        position_counter += 1
                    self.cluster_ids += 1
                                    
        return self.labels
            
    
    def calc_edist_neighbour(self, X: np.ndarray, pt_idx: int):
        difference_matrix = X[pt_idx] - X
        squares = np.square(difference_matrix)
        add_squares = np.sum(squares, axis=-1)
        e_dist = np.sqrt(add_squares)
        satisfy_eps_bool = e_dist <= self.eps
        satisfy_eps_idx = np.flatnonzero(satisfy_eps_bool)
        
        return satisfy_eps_idx

if __name__ == "__main__":
    model = DBSCAN(0.2)

    X = np.array([
    [0.0, 0.0],  # 0
    [0.2, 0.0],  # 1
    [0.0, 0.2],  # 2
    [0.2, 0.2],  # 3
    [0.1, 0.1],  # 4

    [3.0, 3.0],  # 5
    [3.2, 3.0],  # 6
    [3.0, 3.2],  # 7
    [3.2, 3.2],  # 8
    [3.1, 3.1],  # 9

    [7.0, 0.0],]) # isolated
    print(model.fit(X))
    