import numpy as np

class NaiveBayes:
    # Naive Bayes implementation.
    # Assumes the feature are binary.
    # Also assumes the labels go from 0,1,...k-1

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n

        """YOUR CODE HERE FOR Q3.3"""

        # Compute the conditional probabilities i.e.
        # p(x_ij=1 | y_i==c) as p_xy[j, c]
        # p(x_ij=0 | y_i==c) as 1 - p_xy[j, c]
        p_xy = 0.5 * np.ones((d, k))
        # TODO: replace the above line with the proper code 

        # Use counting to populate entries of p_xy
        for j in range(d):
            for c in range(k):
                condition_yes = y == c
                X_yes = X[condition_yes, :]
                y_yes = y[condition_yes]
                p_xy[j, c] = np.mean(X_yes[:, j])  # count the number of times 1 occurs and divide by number of rows in X_yes

        self.p_y = p_y
        self.p_xy = p_xy
        self.not_p_xy = 1 - p_xy

    def predict(self, X):

        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        not_p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(n)
        for i in range(n):

            probs = p_y.copy() # initialize with the p(y) terms
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= not_p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred

class NaiveBayesLaplace(NaiveBayes):
    
    def __init__(self, num_classes, beta=0):
        self.num_classes = num_classes
        self.beta = beta

    def fit(self, X, y):
        """YOUR CODE FOR Q3.4"""
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        # p_y = (counts + self.beta) / (n + self.num_classes * self.beta)
        p_y = counts / n

        # Compute the conditional probabilities with Laplace
        p_xy = 0.5 * np.ones((d, k))
        for j in range(d):
            for c in range(k):
                condition_yes = y == c
                X_yes = X[condition_yes, :]
                y_yes = y[condition_yes]
                n_yes, _ = X_yes.shape
                numerator = np.sum(X_yes[:, j]) + self.beta
                denominator = n_yes + self.num_classes * self.beta
                p_xy[j, c] = numerator / denominator

        not_p_xy = 0.5 * np.ones((d, k))
        for j in range(d):
            for c in range(k):
                condition_yes = y == c
                X_yes = X[condition_yes, :]
                y_yes = y[condition_yes]
                n_yes, _ = X_yes.shape
                numerator = (n_yes - np.sum(X_yes[:, j])) + self.beta
                denominator = n_yes + self.num_classes * self.beta
                not_p_xy[j, c] = numerator / denominator
                
        self.p_y = p_y
        self.p_xy = p_xy
        self.not_p_xy = not_p_xy
    
