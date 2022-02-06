import numpy as np
from utils import euclidean_dist_squared


class Kmeans:
    means = None

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        n, d = X.shape
        y = np.ones(n)

        means = np.zeros((self.k, d))
        for kk in range(self.k):
            i = np.random.randint(n)
            means[kk] = X[i]

        while True:
            # iterations of k-means
            y_old = y

            # Compute euclidean distance to each mean
            distance_matrix = euclidean_dist_squared(X, means)
            distance_matrix[np.isnan(distance_matrix)] = np.inf
            y = np.argmin(distance_matrix, axis=1)

            # Update means
            for kk in range(self.k):
                if np.any(
                    y == kk
                ):  # don't update the mean if no examples are assigned to it (one of several possible approaches)
                    means[kk] = X[y == kk].mean(axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

            # print(self.error(X, y, means))

        self.means = means

    def predict(self, X_hat):
        means = self.means
        distance_matrix = euclidean_dist_squared(X_hat, means)
        distance_matrix[np.isnan(distance_matrix)] = np.inf
        return np.argmin(distance_matrix, axis=1)

    def error(self, X, y, means):
        """YOUR CODE HERE FOR Q5.1"""
        raise NotImplementedError()

