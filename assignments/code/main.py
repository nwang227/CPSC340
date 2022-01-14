#!/usr/bin/env python
import os
from pathlib import Path
import time

# 3rd party libraries
# To make sure you have all of these (but they're all standard):
#    conda install numpy pandas matplotlib-base scipy scikit-learn
# or
#    pip install numpy pandas matplotlib scipy scikit-learn
# Annoyingly, Python's package manager names are not always the same
# as the import names (e.g. scikit-learn vs sklearn).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# CPSC 340 code
from utils import plot_classifier, mode, load_dataset, handle, main, run
import grads
from decision_stump import (
    DecisionStumpEquality,
    DecisionStumpErrorRate,
    DecisionStumpInfoGain,
)
from decision_tree import DecisionTree


@handle("3.4")
def q3_4():
    # Here is some code to test your answers to Q3.4
    # Below we test out example_grad using scipy.optimize.approx_fprime,
    # which approximates gradients.
    # if you want, you can use this to test out your foo_grad and bar_grad

    def check_grad(fun, grad):
        x0 = np.random.rand(5)  # take a random x-vector just for testing
        diff = approx_fprime(x0, fun, 1e-4)  # don't worry about the 1e-4 for now
        print("\n** %s **" % fun.__name__)
        print("My gradient     : %s" % grad(x0))
        print("Scipy's gradient: %s" % diff)

    check_grad(grads.example, grads.example_grad)
    check_grad(grads.foo, grads.foo_grad)
    check_grad(grads.bar, grads.bar_grad)


@handle("5.1")
def q5_1():
    # Load the fluTrends dataset
    df = pd.read_csv(Path("..", "data", "fluTrends.csv"))
    X = df.values
    names = df.columns.values

    # YOUR CODE HERE
    raise NotImplementedError()



@handle("6")
def q6():
    # 1. Load citiesSmall dataset
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]

    # 2. Evaluate majority predictor model
    y_pred = np.zeros(y.size) + mode(y)

    error = np.mean(y_pred != y)
    print("Mode predictor error: %.3f" % error)

    # 3. Evaluate decision stump
    model = DecisionStumpEquality()
    model.fit(X, y)
    y_pred = model.predict(X)

    error = np.mean(y_pred != y)
    print("Decision Stump with equality rule error: %.3f" % error)

    # PLOT RESULT
    plot_classifier(model, X, y)

    fname = Path("..", "figs", "q6_decisionBoundary.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


@handle("6.2")
def q6_2():
    # Load citiesSmall dataset
    dataset = load_dataset("citiesSmall.pkl")
    X = dataset["X"]
    y = dataset["y"]

    # Evaluate decision stump
    model = DecisionStumpErrorRate()
    model.fit(X, y)
    y_pred = model.predict(X)

    error = np.mean(y_pred != y)
    print("Decision Stump with inequality rule error: %.3f" % error)

    # PLOT RESULT
    plot_classifier(model, X, y)

    fname = Path("..", "figs", "q6_2_decisionBoundary.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


@handle("6.3")
def q6_3():
    dataset = load_dataset("citiesSmall.pkl")
    X = dataset["X"]
    y = dataset["y"]

    # Evaluate decision stump
    model = DecisionStumpInfoGain()
    model.fit(X, y)
    y_pred = model.predict(X)

    error = np.mean(y_pred != y)
    print("Decision Stump with info gain rule error: %.3f" % error)

    # PLOT RESULT
    plot_classifier(model, X, y)

    fname = Path("..", "figs", "q6_3_decisionBoundary.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


@handle("6.4")
def q6_4():
    # Load citiesSmall dataset
    dataset = load_dataset("citiesSmall.pkl")
    X = dataset["X"]
    y = dataset["y"]

    # 2. Evaluate decision tree
    model = DecisionTree(max_depth=2, stump_class=DecisionStumpInfoGain)
    model.fit(X, y)

    y_pred = model.predict(X)
    error = np.mean(y_pred != y)

    print(f"Error: {error:.3f}")

    plot_classifier(model, X, y)

    fname = Path("..", "figs", "q6_4_decisionBoundary.pdf")
    plt.savefig(fname)
    print(f"\nFigure saved as {fname}")

    def print_stump(stump):
        print(
            f"Splitting on feature {stump.j_best} at threshold {stump.t_best:f}. "
            f">: {stump.y_hat_yes}, <=: {stump.y_hat_no}"
        )

    print("Top:")
    print_stump(model.stump_model)
    print(">")
    print_stump(model.submodel_yes.stump_model)
    print("<=")
    print_stump(model.submodel_no.stump_model)


@handle("6.5")
def q6_5():
    dataset = load_dataset("citiesSmall")
    X = dataset["X"]
    y = dataset["y"]
    print(f"n = {X.shape[0]}")

    depths = np.arange(1, 15)  # depths to try

    t = time.time()
    my_tree_errors = np.zeros(depths.size)
    for i, max_depth in enumerate(depths):
        # model = DecisionTree(max_depth=max_depth,stump_class=DecisionStumpEquality)
        model = DecisionTree(max_depth=max_depth)
        model.fit(X, y)
        y_pred = model.predict(X)
        my_tree_errors[i] = np.mean(y_pred != y)
    print(
        f"Our decision tree with DecisionStumpErrorRate took {time.time() - t} seconds"
    )

    plt.plot(depths, my_tree_errors, label="errorrate")

    t = time.time()
    my_tree_errors_infogain = np.zeros(depths.size)
    for i, max_depth in enumerate(depths):
        model = DecisionTree(max_depth=max_depth, stump_class=DecisionStumpInfoGain)
        model.fit(X, y)
        y_pred = model.predict(X)
        my_tree_errors_infogain[i] = np.mean(y_pred != y)
    print(
        f"Our decision tree with DecisionStumpInfoGain took {time.time() - t} seconds"
    )

    plt.plot(depths, my_tree_errors_infogain, label="infogain")

    t = time.time()
    sklearn_tree_errors = np.zeros(depths.size)
    for i, max_depth in enumerate(depths):
        model = DecisionTreeClassifier(
            max_depth=max_depth, criterion="entropy", random_state=1
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        sklearn_tree_errors[i] = np.mean(y_pred != y)
    print(f"scikit-learn's decision tree took {time.time() - t} seconds")

    plt.plot(depths, sklearn_tree_errors, label="sklearn", linestyle=":", linewidth=3)

    plt.xlabel("Depth of tree")
    plt.ylabel("Classification error")
    plt.legend()
    fname = Path("..", "figs", "q6_5_tree_errors.pdf")
    plt.savefig(fname)

    # plot the depth 15 sklearn classifier
    model = DecisionTreeClassifier(max_depth=15, criterion="entropy", random_state=1)
    model.fit(X, y)
    plot_classifier(model, X, y)
    fname = Path("..", "figs", "q6_5_decisionBoundary.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


if __name__ == "__main__":
    main()
