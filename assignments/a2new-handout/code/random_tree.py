from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

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
    pass

