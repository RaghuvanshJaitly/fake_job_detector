import numpy as np

class DBSCAN:

    def __init__(self, eps: float, min_samples=5):
        #distance between points/neighbours
        self.eps = eps
        self.min_samples = min_samples
        self.cluster_ids = None
    
    def fit(self, X: np.ndarray):
        #starts as: [-2,-2,-2,-2,-2]
        #outer iter 1 ends: [-1,-2,-2,-2,-2]
        #outer iter2, before while: [-1, 0, -2, -2, -2]
        #outer iter 2, inner iter 1 ends: [0, 0, -2, -2, -2]
        #outer iter 2, inner iter 2 ends: [0, 0, -2, -2, -2]
        #outer iter 2, inner iter 3 ends: [0, 0, 0, -2, -2]
        ##outer iter 2, inner iter 4 ends: [0, 0, 0, 0, -2]
        self.labels = np.full(X.shape[0], -2)
        #starts as: [False, False, False, False, False]
        #iter1: [True, False, False, False, False]
        #iter2: [True, True, False, False, False]
        #iter 2: inner iter1 [True, True, False, False, False]
        #iter 2: inner iter2 [True, True, False, False, False]
        #iter 2: inner iter3 [True, True, True, False, False]
        #iter 2: inner iter4 [True, True, True, True, False]
        self.visited = np.full(X.shape[0], False)
        #starts as : 0
        self.cluster_ids = 0
        #len(x) = 5, so it runs 5 times
        for i in range(len(X)):
            #iter 1: marks [0,0] as visited
            #iter 2: marks [1, 0] as visited
            if not(self.visited[i]):
                self.visited[i] = True
                #iter 1(0): helper caclulates e dist of (0,0) with all other points including itself and returns 
                #the idx of pts that satisfy eps, in this case it returns [0, 1] -> [0,0] and [1,0]
                
                #iter 2(1): helper caclulates e dist of (1,0) with all other points including itself and returns 
                #the idx of pts that satisfy eps, in this case it returns [0, 1, 2] -> [0,0], [1,0] and [2,0]
                satisfy_eps_idx = self.calc_edist_neighbour(X, i)
                #iter 1: True, len(satisfy_eps_idx) is 2, we set (0,0) to noise for now with -1, and we go back to the top for iter 2
                #iter 2: False, len(satisfy_eps_ids) is 3, so we jump to else
                if len(satisfy_eps_idx) < self.min_samples:
                    self.labels[i] = -1
                else:
                    #iter2: we set labels[1] to 0 because cluster_id is 0 rn
                    self.labels[i] = self.cluster_ids
                    #iter 2: worklist = [0, 1 , 2]
                    # after  #iter 2, inner iter 3, inner for iter 2: [0, 1, 2, 3]
                    worklist = satisfy_eps_idx.tolist()
                    #iter 2, inner iter 1: starts as 0
                    # iter 2, inner iter 1
                    position_counter = 0
                    # iter 2, inner iter 1: 0 < len(worklist) which is 0 < 3, so we enter the while loop
                    # iter 2, inner iter 2: 1 < len(worklist) 1 < 3 so while loop goes on
                    # iter 2, inner iter 3: 2 < len(worklist) 2 < 3, so while loop goes on
                    # iter 2, inner iter 4: 3 < len(worklist) 3 < 4, so while loop goes on
                    # iter 2, inner iter 5 check: 4 < len(worklist) 4 < 4, so while loop doesnt run
                    while position_counter < len(worklist):
                        #iter2, inner iter1: curr_idx = 0, which is worklist[0]. = 0
                        #iter2, inner iter 2: curr_idx = 1
                        #iter2, inner iter 3: curr_idx = 2
                        # iter 2, inner iter 4: curr_idx = 3
                        curr_idx = worklist[position_counter]
                        # iter2, inner iter 1: self.visited[0] is True, so we skip the whole if block
                        #iter 2, inner iter 2: self.visited[1] is True, so we skip the whole if block
                        #iter 2, inner iter 3: self.visited[2] is False, so we enter the if block
                        #iter 2, inner iter 4: self.visited[3] is False, so we enter the if block
                        if not(self.visited[curr_idx]):
                            #iter 2, inner iter 3: we set self.visited[2] to True
                            #iter 2, inner iter 4: we set self.visited[3] to True
                            self.visited[curr_idx] = True
                            #iter 2, inner iter 3: helper caclulates e dist of (2,0) with all other points including itself and returns 
                #the idx of pts that satisfy eps, in this case it returns [1, 2, 3] -> [1,0], [2,0] and [3,0]
                            #iter 2, inner iter 3: neighbours = [1,2,3]
                            #iter 2, inner iter 4: neighbours = [2,3]
                            neighbours = self.calc_edist_neighbour(X, curr_idx)
                            #len(neighbours) is 3 which is = min_samples which is 3 as well, so we go in the for loop
                            #len(neighbours) is 2 which is not greater than or equal to 3 so we skip the for loop
                            if len(neighbours) >= self.min_samples:
                                #iter 2, inner iter 3, inner for iter 0:
                                for j in range(len(neighbours)):
                                    #iter 2, inner iter 3, inner for iter 0: neighbours[0] is 1, which is in worklist so we continue
                                    #iter 2, inner iter 3, inner for iter 1: neighbours[1] is 2, which is in worklist so we continue
                                    #iter 2, inner iter 3, inner for iter 2: neighbours [2] is 3, which is not in worklist
                                    if neighbours[j] not in worklist:
                                        #iter 2, inner iter 3, inner for iter 2: we append 3 to worklist
                                        worklist.append(neighbours[j])
                        # iter 2, inner iter 1: self.labels[0] is -1, so we set self.labels[0] = 0
                        # iter2, inner iter 2: self.labels[1] is 0, so we dont do anything
                        #iter2, inner iter 3: self.labels[2] is -2, so we set self.labels[2] = 0
                        #iter 2, inner iter 4: self.labels[3] is -2, so we set self.labels[3] = 0
                        if self.labels[curr_idx] == -1 or self.labels[curr_idx] == -2:
                            self.labels[curr_idx] = self.cluster_ids
                        #iter2, inner iter 1: pos counter = 1
                        #iter 2, inner iter 2: pos counter = 2
                        #iter 2, inner iter 3: pos counter = 3
                        #iter 2, inner iter 4: pos counter = 4
                        position_counter += 1
                    #iter 2 ends, cluster id = 1
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
    model = DBSCAN(1.1, 3)
    X = np.array([
        [0,0],
        [1,0],
        [2,0],
        [3,0],
        [7,0]]) # isolated
    print(model.fit(X))
    print(model.calc_edist_neighbour(X, 3))