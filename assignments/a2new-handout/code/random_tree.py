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
    def __init__(self, max_depth, num_trees):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )
        self.num_trees = num_trees

    def fit(self, X, y):
        #For simplicity, store 50 tree models
       self.trees = []
       for i in range(self.num_trees):
           tree = RandomTree(max_depth=self.max_depth)
           tree.fit(X, y)
           self.trees.append(tree)
            
    def predict(self, X):
        n = X.shape[0]
        y = np.zeros(n)
        result = np.zeros((n, self.num_trees))

        #For each tree, generate a prediction
        for i, tree in enumerate(self.trees):
            y_i  = tree.predict( X)
            result[:,i] = y_i

        #Find the mode of the predictions
        for i in range(n):
            y[i] = utils.mode(result[i,:])

        return y

   

