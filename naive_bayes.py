import numpy as np
import matplotlib.pyplot as plt
class NaiveBayes:
    
    def __init__(self):
        self.mean_ = np.array([])
        self.var_ = np.array([])
        self.prior_prob = np.array([])
        self.classes = np.array([])
        self.score = np.array([])
        
    def fit(self, X:np.ndarray, y:np.ndarray):
        self.classes = np.unique(y)
        #Create empty arrays for mean and variance
        self.mean_ = np.empty((self.classes.shape[0], X.shape[1]))
        self.var_= np.empty((self.classes.shape[0], X.shape[1]))
        self.prior_prob = np.empty((self.classes.shape[0],))
        #Loop through y and calculate the mean and variance to store in the specific arrays
        for i, class_label in enumerate(self.classes):
            mask = y == class_label
            #current class for eg: "A" or "B"
            curr_class = X[mask]
            curr_class_mean = np.mean(curr_class, axis=0)
            curr_class_var = np.var(curr_class, axis=0)
            curr_prior_prob = len(curr_class)/len(y)
            self.mean_[i] = curr_class_mean
            self.var_[i] = curr_class_var
            self.prior_prob[i] = curr_prior_prob
    
    def predict(self, X: np.ndarray):
        self.score = np.empty((self.classes.shape[0],))
        for i, class_label in enumerate(self.classes):
            prob = (1/np.sqrt(2 * np.pi * self.var_[i])) * np.exp(-((X - self.mean_[i])**2)/(2 * self.var_[i]))
            all_prob = np.prod(prob)
            self.score[i] = all_prob * self.prior_prob[i]

        res = self.score.max()
        max_idx = np.argmax(self.score)
        return res, self.classes[max_idx]
    
if __name__ == "__main__":
    X = np.array([
  [1.4, 0.2],
  [1.5, 0.2],
  [1.3, 0.2],
  [1.6, 0.3],
  [1.4, 0.3],
  [1.7, 0.4],

  [4.7, 1.4],
  [4.5, 1.5],
  [4.9, 1.5],
  [4.0, 1.3],
  [4.6, 1.5],
  [4.5, 1.3]])
    
    y = np.array([ "A", "A", "A", "A", "A", "A","B", "B", "B", "B", "B", "B"])
    
    model = NaiveBayes()
    model.fit(X,y)
    x_test = np.array([3.0, 0.85])
    res, winner = model.predict(x_test)
    print(res)
    print(winner)