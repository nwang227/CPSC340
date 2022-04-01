"""
Implementation of k-nearest neighbours classifier
"""

import utils
from utils import euclidean_dist_squared
import numpy as np
from scipy import stats

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, X_hat):
        """YOUR CODE HERE FOR Q2"""
        t, d = X_hat.shape
        y_hat = np.zeros(t, dtype=np.uint8)
        
        # Compute n-by-t distance matrix. Sometimes called "D".
        distance_matrix = euclidean_dist_squared(self.X, X_hat)

        # Iterate through rows of distance matrix
        for i in range(t):
            distances_from_x_hat_i = distance_matrix[:, i]
            nn_idxs = np.argsort(distances_from_x_hat_i)
            knn_idxs = nn_idxs[:self.k]
            y_knns = self.y[knn_idxs]
            y_hat[i] = utils.mode(y_knns)

        return y_hat

