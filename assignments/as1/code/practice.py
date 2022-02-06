from ctypes import util
from turtle import register_shape
import pandas as pd
import os
import numpy as np
from utils import plot_classifier, mode, load_dataset, handle, main, run
from pathlib import Path
import matplotlib.pyplot as plt
# from decision_stump import (
#     DecisionStumpEquality,
#     DecisionStumpErrorRate)

os.chdir("/Users/nanwang/Desktop/MachineLearning/CPSC340/assignments/code")


dataset = load_dataset("citiesSmall.pkl")
X = dataset["X"]
y = dataset["y"]

less = X[:,1] < 30
y_yes = y[less]
y_no = y[~less]
p_yes = np.bincount(y_yes)/y_yes.size
p_no = np.bincount(y_no)/y_no.size

print(p_yes,p_no)
