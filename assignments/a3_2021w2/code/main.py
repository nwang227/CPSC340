#!/usr/bin/env python
import argparse
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


# this just some Python scaffolding to conveniently run the functions below;
# don't worry about figuring out how it works if it's not obvious to you
_funcs = {}


def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(question):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    return _funcs[question]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", required=True, choices=_funcs.keys())
    args = parser.parse_args()
    return run(args.question)


@handle("2")
def q2():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    # Fit least-squares estimator
    model = linear_models.LeastSquares()
    model.fit(X, y)
    print(model.w)

    utils.test_and_plot(
        model, X, y, title="Least Squares", filename="least_squares_outliers.pdf"
    )


@handle("2.1")
def q2_1():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    """YOUR CODE FOR Q2.1"""
    raise NotImplementedError()


@handle("2.4")
def q2_4():
    # loads the data in the form of dictionary
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    fun_obj = FunObjLeastSquares()
    optimizer = OptimizerGradientDescentLineSearch(max_evals=100, verbose=False)
    model = linear_models.LinearModelGradientDescent(fun_obj, optimizer)
    model.fit(X, y)
    print(model.w)

    utils.test_and_plot(
        model,
        X,
        y,
        title="Linear Regression with Gradient Descent",
        filename="least_squares_gd.pdf",
    )


@handle("2.4.1")
def q2_4_1():
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    """YOUR CODE HERE FOR Q2.4.1"""
    # TODO: Finish FunObjRobustRegression in fun_obj.py.
    raise NotImplementedError()


@handle("2.4.2")
def q2_4_2():
    # loads the data in the form of dictionary
    data = load_dataset("outliersData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)

    # Produce the learning curves with
    # 1. OptimizerGradientDescent
    # 2. OptimizerGradientDescentLineSearch

    fun_obj = FunObjRobustRegression()
    optimizer = OptimizerGradientDescent(max_evals=100, verbose=False)
    model = linear_models.LinearModelGradientDescent(fun_obj, optimizer)
    model.fit(X, y)
    """YOUR CODE HERE FOR Q2.4.2"""
    raise NotImplementedError()


@handle("3")
def q3():
    data = load_dataset("basisData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    X_valid = data["Xtest"]
    y_valid = data["ytest"].squeeze(1)

    # Fit least-squares model
    model = linear_models.LeastSquares()
    model.fit(X, y)

    utils.test_and_plot(
        model,
        X,
        y,
        X_valid,
        y_valid,
        title="Least Squares, no bias",
        filename="least_squares_no_bias.pdf",
    )


@handle("3.1")
def q3_1():
    data = load_dataset("basisData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    X_valid = data["Xtest"]
    y_valid = data["ytest"].squeeze(1)

    model = linear_models.LeastSquaresBias()
    model.fit(X, y)

    utils.test_and_plot(
        model,
        X,
        y,
        X_valid,
        y_valid,
        title="Least Squares, yes bias",
        filename="least_squares_yes_bias.pdf",
    )


@handle("3.2")
def q3_2():
    data = load_dataset("basisData.pkl")
    X = data["X"]
    y = data["y"].squeeze(1)
    X_valid = data["Xtest"]
    y_valid = data["ytest"].squeeze(1)

    p_vals = [0,1,2,3,4,5,10,20,30,50,75,100]
    num_runs = len(p_vals)
    err_trains = np.zeros(num_runs)
    err_valids = np.zeros(num_runs)

    plot_grid_size1 = int(np.ceil(np.sqrt(num_runs)))
    plot_grid_size2 = int(np.ceil(num_runs / plot_grid_size1))

    fig, axes = plt.subplots(
        plot_grid_size1, plot_grid_size2, figsize=(30, 20),
        sharex=True, sharey=True, constrained_layout=True
    )
    for i, (p, ax) in enumerate(zip(p_vals, (ax for row in axes for ax in row))):
        print(f"p = {p}")

        model = linear_models.LeastSquaresPoly(p)
        model.fit(X, y)
        y_hat = model.predict(X)
        err_train = np.mean((y_hat - y) ** 2)
        err_trains[i] = err_train

        y_hat = model.predict(X_valid)
        err_valid = np.mean((y_hat - y_valid) ** 2)
        err_valids[i] = err_valid

        ax.scatter(X, y, color="b", s=2)
        Xgrid = np.linspace(np.min(X_valid), np.max(X_valid), 1000)[:, None]
        ygrid = model.predict(Xgrid)
        ax.plot(Xgrid, ygrid, color="r")
        ax.set_title(f"p={p}")
        ax.set_ylim(np.min(y), np.max(y))

    filename = Path("..", "figs", "polynomial_fits.pdf")
    print("Saving to", filename)
    fig.savefig(filename)

    # Plot error curves
    plt.figure()
    plt.plot(p_vals, err_trains, marker='o', label="training error")
    plt.plot(p_vals, err_valids, marker='o', label="validation error")
    plt.xlabel("Degree of polynomial")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.legend()
    filename = Path("..", "figs", "polynomial_error_curves.pdf")
    print("Saving to", filename)
    plt.savefig(filename)


if __name__ == "__main__":
    main()
