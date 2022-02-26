#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from fun_obj import (
    LogisticRegressionLoss,
    LogisticRegressionLossL0,
    LogisticRegressionLossL2,
    SoftmaxLoss
)
import linear_models
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    GradientDescentLineSearchProxL1,
)
import utils

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
    data = utils.load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    fun_obj = LogisticRegressionLoss()
    optimizer = GradientDescentLineSearch(max_evals=400, verbose=False)
    model = linear_models.LogRegClassifier(fun_obj, optimizer)
    model.fit(X, y)

    train_err = utils.classification_error(model.predict(X), y)
    print(f"LogReg Training error: {train_err:.3f}")

    val_err = utils.classification_error(model.predict(X_valid), y_valid)
    print(f"LogReg Validation error: {val_err:.3f}")

    print(f"# nonZeros: {np.sum(model.w != 0)}")
    print(f"# function evals: {optimizer.num_evals}")


@handle("2.1")
def q2_1():
    data = utils.load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    fun_obj = LogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch(max_evals=400, verbose=False)
    model = linear_models.LogRegClassifier(fun_obj, optimizer)
    model.fit(X, y)

    train_err = utils.classification_error(model.predict(X), y)
    print(f"LogReg Training error: {train_err:.3f}")

    val_err = utils.classification_error(model.predict(X_valid), y_valid)
    print(f"LogReg Validation error: {val_err:.3f}")

    print(f"# nonZeros: {np.sum(model.w != 0)}")
    print(f"# function evals: {optimizer.num_evals}")


@handle("2.2")
def q2_2():
    data = utils.load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    """YOUR CODE HERE FOR Q2.2"""
    raise NotImplementedError()


@handle("2.3")
def q2_3():
    data = utils.load_dataset("logisticData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    local_loss = LogisticRegressionLoss()
    global_loss = LogisticRegressionLossL0(1)
    optimizer = GradientDescentLineSearch(max_evals=400, verbose=False)
    model = linear_models.LogRegClassifierForwardSel(local_loss, global_loss, optimizer)
    model.fit(X, y)

    train_err = utils.classification_error(model.predict(X), y)
    print(f"LogReg training 0-1 error: {train_err:.3f}")

    val_err = utils.classification_error(model.predict(X_valid), y_valid)
    print(f"LogReg validation 0-1 error: {val_err:.3f}")

    print(f"# nonZeros: {np.sum(model.w != 0)}")
    print(f"total function evaluations: {model.total_evals:,}")


@handle("3")
def q3():
    data = utils.load_dataset("multiData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    model = linear_models.LeastSquaresClassifier()
    model.fit(X, y)


    train_err = utils.classification_error(model.predict(X), y)
    print(f"LeastSquaresClassifier training 0-1 error: {train_err:.3f}")

    val_err = utils.classification_error(model.predict(X_valid), y_valid)
    print(f"LeastSquaresClassifier validation 0-1 error: {val_err:.3f}")

    print(f"model predicted classes: {np.unique(model.predict(X))}")


@handle("3.2")
def q3_2():
    data = utils.load_dataset("multiData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    fun_obj = LogisticRegressionLoss()
    optimizer = GradientDescentLineSearch(max_evals=500, verbose=False)
    model = linear_models.LogRegClassifierOneVsAll(fun_obj, optimizer)
    model.fit(X, y)

    train_err = utils.classification_error(model.predict(X), y)
    print(f"LogRegClassifierOneVsAll training 0-1 error: {train_err:.3f}")

    val_err = utils.classification_error(model.predict(X_valid), y_valid)
    print(f"LogRegClassifierOneVsAll validation 0-1 error: {val_err:.3f}")

    print(f"model predicted classes: {np.unique(model.predict(X))}")


@handle("3.4")
def q3_4():
    data = utils.load_dataset("multiData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    fun_obj = SoftmaxLoss()
    optimizer = GradientDescentLineSearch(max_evals=1_000, verbose=True)
    model = linear_models.MulticlassLogRegClassifier(fun_obj, optimizer)
    model.fit(X, y)

    train_err = utils.classification_error(model.predict(X), y)
    print(f"SoftmaxLoss training 0-1 error: {train_err:.3f}")

    val_err = utils.classification_error(model.predict(X_valid), y_valid)
    print(f"SoftmaxLoss validation 0-1 error: {val_err:.3f}")

    print(f"model predicted classes: {np.unique(model.predict(X))}")


@handle("3.5")
def q3_5():
    from sklearn.linear_model import LogisticRegression

    data = utils.load_dataset("multiData")
    X, y = data["X"], data["y"]
    X_valid, y_valid = data["Xvalid"], data["yvalid"]

    """YOUR CODE HERE FOR Q3.5"""
    raise NotImplementedError()


if __name__ == "__main__":
    main()
