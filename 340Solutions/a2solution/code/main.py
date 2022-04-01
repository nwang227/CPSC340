# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes, NaiveBayesLaplace

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomForest, RandomTree

from kmeans import Kmeans
from sklearn.cluster import DBSCAN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_hat = model.predict(X)
        err_train = np.mean(y_hat != y)

        y_hat = model.predict(X_test)
        err_test = np.mean(y_hat != y_test)
        print("Training error: {:.3f}".format(err_train))
        print("Testing error: {:.3f}".format(err_test))

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        """YOUR CODE HERE FOR Q1.1"""

        n_depths = 15
        err_trains = []
        err_tests = []
        for depth in np.arange(1, n_depths + 1):
            model = DecisionTreeClassifier(max_depth=depth, criterion='entropy')
            # model = DecisionTree(max_depth=depth, stump_class=DecisionStumpInfoGain)

            model.fit(X, y)
            
            y_hat = model.predict(X)
            err_train = np.mean(y_hat != y)

            y_hat = model.predict(X_test)
            err_test = np.mean(y_hat != y_test)

            err_trains.append(err_train)
            err_tests.append(err_test)

        plt.figure()
        plt.plot(err_trains, label="training error")
        plt.plot(err_tests, label="testing error")
        plt.xlabel("Depth")
        plt.ylabel("Error")
        plt.legend()
        fname = os.path.join("..", "figs", "trainTest_mine.pdf")
        plt.savefig(fname)
        print("Figure saved as {:s}".format(fname))



    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape

        """YOUR CODE HERE FOR Q1.2"""
        idx_train = np.arange(int(n/2))
        idx_valid = np.arange(int(n/2), n)

        # Flip train and valid
        # idx_valid = np.arange(int(n/2))
        # idx_train = np.arange(int(n/2), n)

        X_train = X[idx_train, :]
        y_train = y[idx_train]
        X_valid = X[idx_valid, :]
        y_valid = y[idx_valid]

        n_depths = 15
        best_depth = 0
        best_err_valid = np.inf
        for max_depth in np.arange(1, n_depths + 1):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion="entropy", random_state=1)
            # model = DecisionTree(max_depth=max_depth, stump_class=DecisionStumpInfoGain)
            model.fit(X_train, y_train)

            y_hat = model.predict(X_valid)
            err_valid = np.mean(y_valid != y_hat)

            if err_valid < best_err_valid:
                best_depth = max_depth
                best_err_valid = err_valid

        print("best_depth={:d}, best_err_valid={:.3f}".format(best_depth, best_err_valid))

    
    elif question == '1.3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        n, d = X.shape

        """YOUR CODE HERE FOR Q1.3"""
        idx_valid = np.arange(15)
        idx_train = np.arange(15, n)

        # Flip train and valid
        # idx_train = np.arange(15)
        # idx_valid = np.arange(15, n)

        X_train = X[idx_train, :]
        y_train = y[idx_train]
        X_valid = X[idx_valid, :]
        y_valid = y[idx_valid]

        n_depths = 15
        best_depth = 0
        best_err_valid = np.inf
        for max_depth in np.arange(1, n_depths + 1):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion="entropy", random_state=1)
            model.fit(X_train, y_train)

            y_hat = model.predict(X_train)
            err_train = np.mean(y_train != y_hat)

            y_hat = model.predict(X_valid)
            err_valid = np.mean(y_valid != y_hat)

            if err_valid < best_err_valid:
                best_depth = max_depth
                best_err_valid = err_valid

        model = DecisionTreeClassifier(max_depth=best_depth, criterion="entropy", random_state=1)
        model.fit(X, y)

        y_hat = model.predict(X_test)
        err_test = np.mean(y_test != y_hat)

        print("best_depth={:d}, best_err_valid={:.3f}, err_test={:.3f}".format(best_depth, best_err_valid, err_test))



    elif question == '2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']

        """YOUR CODE HERE FOR Q2"""
        
        # Q2.2: Compute training and test error with different k values
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

        # Q2.3: plot_classifier

        # My implementation
        model = KNN(k=1)
        model.fit(X, y)
        utils.plot_classifier(model, X, y)
        fname = os.path.join("..", "figs", "knnDecisionBoundary.pdf")
        plt.savefig(fname)
        print("Figure saved as {:s}".format(fname))

        # sklearn implementation
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X, y)
        utils.plot_classifier(model, X, y)
        fname = os.path.join("..", "figs", "knnDecisionBoundary_sklearn.pdf")
        plt.savefig(fname)
        print("Figure saved as {:s}".format(fname))

    elif question == '3.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

        """YOUR CODE HERE FOR Q3.2"""
        # Q3.2.1
        print("Column 41 corresponds to: {:s}".format(wordlist[40]))  # prints "honda"

        # Q3.2.2
        # Look at example 400 (401th example)
        # It's supposed to be a boolean vector, but numpy will interpret it as integer indexing vector. Cast it to boolean.
        xi = X[400, :].astype(np.bool)
        print("Example 401 contains these words: ", wordlist[xi])  # prints "christian", "fact", "god", "studies", "university"

        # Q3.2.3
        print("Groupname for 401 is: {:s}".format(groupnames[y[400]])) # prints "talk.*"

    elif question == '3.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
 
        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)

        y_hat = model.predict(X)
        err_train = np.mean(y_hat != y)
        print("Naive Bayes (ours) training error: {:.3f}".format(err_train))
        
        y_hat = model.predict(X_valid)
        err_valid = np.mean(y_hat != y_valid)
        print("Naive Bayes (ours) validation error: {:.3f}".format(err_valid))

    elif question == '3.4':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
 
        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        
        """YOUR CODE HERE FOR Q3.4"""
        print(model.p_xy[:, 0])

        model = NaiveBayesLaplace(num_classes=4, beta=1.0)
        model.fit(X, y)

        print(model.p_xy[:, 0])

        model = NaiveBayesLaplace(num_classes=4, beta=10000.0)
        model.fit(X, y)

        print(model.p_xy[:, 0])

    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

        """YOUR CODE FOR Q4"""
        # Q4.1
        print("Random tree")
        evaluate_model(RandomTree(max_depth=np.inf))

        # Q4.3
        print("Random forest")
        evaluate_model(RandomForest(50, np.inf))

    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic_rerun.png")
        plt.savefig(fname)
        print("Figure saved as {:s}".format(fname))

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']
        
        """YOUR CODE HERE FOR Q5.1"""
        # Q5.1.1
        model = Kmeans(k=4)
        model.fit(X)

        # Q5.1.2
        best_error = np.inf
        best_model = None
        for _ in range(50):
            model = Kmeans(k=4)
            model.fit(X)
            y = model.predict(X)
            err = model.error(X, y, model.means)
            if err < best_error:
                best_error = err
                best_model = model

        print(best_error)
        y = best_model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")
        fname = os.path.join("..", "figs", "kmeans_50_inits.png")
        plt.savefig(fname)
        print("Figure saved as {:s}".format(fname))

    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']

        """YOUR CODE HERE FOR Q5.2"""
        best_errors = []
        ks = np.arange(1, 11)
        for k in ks:
            best_error = np.inf
            for _ in range(50):
                model = Kmeans(k=k)
                model.fit(X)
                y = model.predict(X)
                err = model.error(X, y, model.means)
                if err < best_error:
                    best_error = err
                    best_model = model
            best_errors.append(best_error)
        plt.plot(ks, best_errors)
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.title("k-means training error as k increases")
        fname = os.path.join("..", "figs", "kmeans_err_k.png")
        plt.savefig(fname)
        print("Figure saved as {:s}".format(fname))

    else:
        print("Unknown question: {:s}".format(question))
