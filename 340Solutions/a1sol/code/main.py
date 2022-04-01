# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np                              # this comes with Anaconda
import pandas as pd                             # this comes with Anaconda
import matplotlib.pyplot as plt                 # this comes with Anaconda
from scipy.optimize import approx_fprime        # this comes with Anaconda
from sklearn.tree import DecisionTreeClassifier # if using Anaconda, install with `conda install scikit-learn`
""" NOTE:
Python is nice, but it's not perfect. One horrible thing about Python is that a 
package might use different names for installation and importing. For example, 
seeing code with `import sklearn` you might sensibly try to install the package 
with `conda install sklearn` or `pip install sklearn`. But, in fact, the actual 
way to install it is `conda install scikit-learn` or `pip install scikit-learn`.
Wouldn't it be lovely if the same name was used in both places, instead of 
`sklearn` and then `scikit-learn`? Please be aware of this annoying feature. 
"""

# CPSC 340 code
import utils
import grads
from decision_stump import DecisionStumpEquality, DecisionStumpErrorRate, DecisionStumpInfoGain
from decision_tree import DecisionTree


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "3.4":
        # Here is some code to test your answers to Q3.4
        # Below we test out example_grad using scipy.optimize.approx_fprime, which approximates gradients.
        # if you want, you can use this to test out your foo_grad and bar_grad

        def check_grad(fun, grad):
            x0 = np.random.rand(5) # take a random x-vector just for testing
            diff = approx_fprime(x0, fun, 1e-4)  # don't worry about the 1e-4 for now
            print("\n** %s **" % fun.__name__)
            print("My gradient     : %s" % grad(x0))
            print("Scipy's gradient: %s" % diff)

        check_grad(grads.example, grads.example_grad)
        # check_grad(grads.foo, grads.foo_grad)
        # check_grad(grads.bar, grads.bar_grad)


    elif question == "5.1":
        # Load the fluTrends dataset
        df = pd.read_csv(os.path.join('..','data','fluTrends.csv'))
        X = df.values
        names = df.columns.values

        # part 1: min, max, mean, median and mode
        results = ("Min: %.3f, Max: %.3f, Mean: %.3f, Median: %.3f, Mode: %.3f" %
                   (np.min(X), np.max(X), np.mean(X), np.median(X), utils.mode(X)))
        print(results)

        # part 2: quantiles
        print("quantiles: %s" % np.percentile(X, [5, 25, 50, 75, 95]))

        # part 3: maxMean, minMean, maxVar, minVar
        means = np.mean(X, axis=0)
        variances = np.var(X, axis=0)

        results = ("maxMean: %s, minMean: %s, maxVar: %s, minVar: %s" %
                   (names[np.argmax(means)], names[np.argmin(means)],
                    names[np.argmax(variances)], names[np.argmin(variances)]))
        print(results)

        # part 4: correlation between columns
        corr = np.corrcoef(X.T)
        
        # least correlated
        c_min = np.min(corr)
        c_least = np.where(corr == c_min)[0]
        
        # most correlated
        c_max = np.max(corr - np.identity(X.shape[1]))
        c_most = np.where(corr == c_max)

        print("most correlated is between %s and %s\n"
              "least correlated is between %s and %s" %
              (names[int(c_most[0])], names[int(c_most[1])],
               names[c_least[0]], names[c_least[1]]))

    elif question == "6":
        # 1. Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump
        model = DecisionStumpEquality()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y) 
        print("Decision Stump with equality rule error: %.3f"
              % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "6.2":
        # Q2.2 - Decision Stump

        # 1. Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "6.3":
        # 1. Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with info gain rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_3_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
    
    elif question == "6.4":
        # 1. Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = DecisionTree(max_depth=2,stump_class=DecisionStumpInfoGain)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)
        
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_4_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        
        def print_stump(stump):
            print("Splitting on feature %s at threshold %f. >: %s, <=: %s" % (stump.splitVariable, stump.splitValue, stump.splitSat, stump.splitNot))

        print("Top:")
        print_stump(model.splitModel)
        print(">")
        print_stump(model.subModel1.splitModel)
        print("<=")
        print_stump(model.subModel0.splitModel)
        # print("Top:   splitting on feature %s at threshold %f" % (model.splitModel.splitVariable, model.splitModel.splitValue))
        # print("Left:  splitting on feature %s at threshold %f" % (model.subModel0.splitVariable, model.subModel0.splitValue))
        # print("Right: splitting on feature %s at threshold %f" % (model.subModel1.splitVariable, model.subModel1.splitValue))


    elif question == "6.5":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        # 
        # t = time.time()
        # equality_tree_errors = np.zeros(depths.size)
        # for i, max_depth in enumerate(depths):
        #     model = DecisionTree(max_depth=max_depth,stump_class=DecisionStumpEquality)
        #     model.fit(X, y)
        #     y_pred = model.predict(X)
        #     equality_tree_errors[i] = np.mean(y_pred != y)
        # print("Decision tree with DecisionStumpEquality took %f seconds" % (time.time()-t))
        # 
        # plt.plot(depths, equality_tree_errors, label="equality")
        # 
        
        
        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            # model = DecisionTree(max_depth=max_depth,stump_class=DecisionStumpEquality)
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("Our decision tree with DecisionStumpErrorRate took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors, label="errorrate")
        
        
        t = time.time()
        my_tree_errors_infogain = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth,stump_class=DecisionStumpInfoGain)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors_infogain[i] = np.mean(y_pred != y)
        print("Our decision tree with DecisionStumpInfoGain took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors_infogain, label="infogain")

        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            sklearn_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

        plt.plot(depths, sklearn_tree_errors, label="sklearn", linestyle=":", linewidth=3)

        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q6_5_tree_errors.pdf")
        plt.savefig(fname)
        
        # plot the depth 15 sklearn classifier
        model = DecisionTreeClassifier(max_depth=15, criterion='entropy', random_state=1)
        model.fit(X, y)
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q6_5_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    else:
        print("No code to run for question", question)