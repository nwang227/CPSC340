from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np
import utils

class RandomTree(DecisionTree):
        
    def __init__(self, max_depth):
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpInfoGain)

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]
        
        DecisionTree.fit(self, bootstrap_X, bootstrap_y)

class RandomForest:
    """
    YOUR CODE HERE FOR Q4
    Hint: start with the constructor __init__(), which takes the hyperparameters.
    Hint: you can instantiate objects inside fit().
    Make sure predict() is able to handle multiple examples.
    """
    
    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for c in range(self.num_trees):
            tree = RandomTree(max_depth=self.max_depth)
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X_hat):
        t, d = X_hat.shape
        pred_matrix = np.zeros([t, self.num_trees])  #  each row is a num_trees-by-1 vector corresponding to example
        for c, tree in enumerate(self.trees):
            y_pred_for_tree = tree.predict(X_hat)  #  t-by-1 vector
            pred_matrix[:, c] = y_pred_for_tree  # set as column for tree
        y_pred = np.zeros(t, dtype=np.uint8)
        for i in range(t):
            y_pred[i] = utils.mode(pred_matrix[i, :])
        return y_pred
