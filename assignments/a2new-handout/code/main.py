#!/usr/bin/env python
import argparse
from cProfile import label
import os
import pickle
from pathlib import Path
from random_stump import RandomStumpInfoGain
import utils

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    """YOUR CODE HERE FOR Q1"""
    #Q1.2 Report training and testing error
    ks = np.array([1, 3, 10])
    for k in ks:
        model = KNN(k=k)
        model.fit(X, y)

        # Compute training error
        y_hat = model.predict(X)
        err_train = np.mean(y != y_hat)
            
        # Compute test error
        y_hat = model.predict(X_test)
        err_test = np.mean(y_test != y_hat)

        print("k={:d}, err_train={:.3f}, err_test={:.3f}".format(k, err_train, err_test))

    #Q1.3 Plot figure
    model = KNN(k = 1)
    model.fit(X,y)
    utils.plot_classifier(model, X, y)
    fname = os.path.join("..", "figs", "knnDecisionBoundary.pdf")
    plt.savefig(fname)
    print("Figure saved as {:s}".format(fname))






@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    n,d = X.shape

    ks = list(range(1, 30, 4))
    train = np.zeros((8,10))
    test = np.zeros((8,10))
    result = np.zeros((8,4))

    for  a in range(1,9,1):
        k_i = (a - 1) * 4 + 1 
        result[a-1,0] = k_i
        
        #Compute testing error
        model = KNN(k = k_i)
        model.fit(X, y)
        y_hat = model.predict(X_test)
        err = np.mean(y_test != y_hat)
        result[a-1,3] = err

        #Compute errors with cross-validation
        for b in range(1,11,1):
            low = int(0.1*(b-1)*n)
            high = int(0.1*b*n)

            #Generate Mask Array
            mask = np.ones(n, dtype=bool)
            mask[low:high] = False

            #Use Mask to Select Data
            X_test_i = X[~mask,:]
            y_test_i = y[~mask]
            X_i = X[mask,:]
            y_i = y[mask]

            #Compute Errors
            model = KNN(k = k_i)

            #Compute train error
            model.fit(X_i, y_i)
            y_hat = model.predict(X_i)
            err_train = np.mean(y_hat != y_i)
            train[a-1,b-1] = err_train
            
            # Compute test error
            y_hat_test = model.predict(X_test_i)
            err_test = np.mean(y_hat_test != y_test_i)
            test[a-1,b-1] = err_test


        result[a-1, 1] = np.mean(train[a-1,:])
        result[a-1, 2] = np.mean(test[a-1,:])
    cv_accs = result[:,2]

    #Plot Cross-Validation
    plt.plot(result[:,0], result[:,3], label = "Testing Error")
    plt.plot(result[:,0], result[:,2], label = "Cross-Validation")
    plt.legend()
    plt.xlabel("K-level")
    plt.ylabel("Error") 
    plt.title("Cross-Validation")
    fname1 = os.path.join("..", "figs", "Cross-Validation.pdf")
    plt.savefig(fname1)


    #Plot Training Errors
    plt.clf()
    plt.plot(result[:,0], result[:,1], label = "Training Error")
    plt.legend()
    plt.xlabel("K-level")
    plt.ylabel("Error")
    plt.title("Training Error")
    fname2 = os.path.join("..", "figs", "TrainingError.pdf")
    plt.savefig(fname2)








@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    word73 = wordlist[72]
    mask = X[802,:]
    word803 = wordlist[mask]
    group803 = y[802]

    print(word73, word803, group803)




@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")

@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    for beta in {1,10000}:
        model = NaiveBayesLaplace(num_classes=4)

        model.fit(X, y, beta)
        print(f"beta = {beta}")

        y_hat = model.predict(X)
        err_train = np.mean(y_hat != y)
        print(f"Naive Bayes Laplace training error: {err_train:.3f}")

        y_hat = model.predict(X_valid)
        err_valid = np.mean(y_hat != y_valid)
        print(f"Naive Bayes Laplace validation error: {err_valid:.3f}")


@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    print("Random Forest")
    evaluate_model(RandomTree(max_depth=np.inf))

    print("Random Forest")
    evaluate_model(RandomForest(max_depth=np.inf, num_trees = 50))
    pass



@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")



@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]
    n = X.shape[0]
    t = 50
    errors = np.zeros(t)

    for i in range(t):
        model = Kmeans(k = 4)
        model.fit(X)
        y = model.predict(X)
        errors[i] = model.error(X,y, model.means)
    low_err = np.min(errors)
    print(low_err)

    plt.plot(errors)
    plt.title("Change of Error")
    fname = os.path.join("..", "figs", "kmeans_error.pdf")
    plt.savefig(fname)



@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]
    
    n = X.shape[0]
    t = 50
    errors = np.zeros(t)
    result = np.zeros((n,t))
    best_error = np.zeros((10,2))

    for kk in range(1,11,1):
        for i in range(t):
            model = Kmeans(k = kk)
            model.fit(X)
            y = model.predict(X)
            errors[i] = model.error(X,y, model.means)
        best_error[kk-1,0] = np.min(errors)
        best_error[kk-1,1] = kk

    plt.plot(best_error[:,1], best_error[:,0], label = "Error")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.title("Change of Error over k")
    fname = os.path.join("..", "figs", "k_selection.pdf")
    plt.savefig(fname)
       

q5_2()

 

    


if __name__ == "__main__":
    main()
