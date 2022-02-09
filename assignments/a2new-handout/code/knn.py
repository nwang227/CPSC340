"""
Implementation of k-nearest neighbours classifier
"""

from turtle import end_fill
import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        n,d = X_hat.shape
        y_hat = np.zeros(n)

        dist = euclidean_dist_squared(self.X, X_hat)
        
        for i in range(n):
            dist_x_hat_i = dist[:,i]
            order = np.argsort(dist_x_hat_i)
            select = order[:self.k]
            KNN_i = self.y[select]
            y_hat[i] = utils.mode(KNN_i)
        return y_hat


