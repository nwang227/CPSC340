import numpy as np

class PCA:
    """
    Solves the PCA problem min_Z,W (Z*W-X)^2 using SVD
    """

    def __init__(self, k):
        self.k = k
        self.mu = None
        self.W = None

    def fit(self, X):
        """
        Learns the principal components by delegating to SVD solver.
        "Fitting" here is the matter of populating:
        self.mu: the column-wise mean
        self.W: the principal components
        """
        self.mu = np.mean(X,axis=0)
        X = X - self.mu

        U, s, Vh = np.linalg.svd(X)
        self.W = Vh[:self.k]

    def compress(self, X):
        """
        Use the column-wise mean and principal components to
        compute the "component scores" to compress
        """
        X = X - self.mu
        Z = X@self.W.T
        return Z

    def expand(self, Z):
        X = Z@self.W + self.mu
        return X

