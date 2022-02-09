import argparse
import os
import pickle
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


os.chdir(Path(__file__).parent.resolve())


dataset = load_dataset("newsgroups.pkl")

X = dataset["X"]
y = dataset["y"]
X_valid = dataset["Xvalidate"]
y_valid = dataset["yvalidate"]
groupnames = dataset["groupnames"]
wordlist = dataset["wordlist"]
n, d = X.shape
counts = np.bincount(y)
p_y = counts / n

k = 4

p_xy = np.zeros((d, k))

for j in range(k):
    mask_j = np.where(y == j , True, False)
    X_j = X[~mask_j]
    for i in range(d):
        p_xy[i,j] = np.mean(X_j[:,i])
        print(p_xy)

