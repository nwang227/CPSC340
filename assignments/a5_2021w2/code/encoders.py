import numpy as np

"""
Contains class definitions related to latent factor models, whose behaviours are
encapsulated by the "learned encoders", which are objects implementing encode() method.
"""


class LinearEncoder:
    """
    Latent factor models that can "encode" X into Z, and "decode" Z into X based on latent factors W.
    """

    mu = None
    W = None

    def encode(self, X):
        """
        Use the column-wise mean and principal components to
        compute the "component scores" to encode
        """
        X = X - self.mu
        return X @ self.W.T

    def decode(self, Z):
        """
        Transforms "component scores" back into the original data space.
        """
        return Z @ self.W + self.mu


class PCAEncoder(LinearEncoder):
    """
    Solves the PCA problem min_{Z,W} (Z*W - X)^2 using SVD
    """

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        """
        Learns the principal components by delegating to SVD solver.
        "Fitting" here is the matter of populating:
        self.mu: the column-wise mean
        self.W: the principal components
        """
        self.mu = np.mean(X, axis=0)
        X = X - self.mu

        U, s, Vh = np.linalg.svd(X)
        self.W = Vh[: self.k]
        self.X = X
