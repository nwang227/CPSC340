import numpy as np
import argparse
from cProfile import label
import os
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from optimizers import OptimizerGradientDescent, OptimizerGradientDescentLineSearch
from fun_obj import FunObjLeastSquares, FunObjRobustRegression
import linear_models
import utils


def load_dataset(filename):
    with open(Path("..", "data", filename), "rb") as f:
        return pickle.load(f)


data = load_dataset("basisData.pkl")
X = data["X"]

n,d = X.shape
Z_old = np.ones((n,1))
Z_new = Z_old
for i in range(3):
    Z_new = np.append(Z_old, np.power(X,i+1), axis = 1)
    Z_old = Z_new
print(Z_new)

