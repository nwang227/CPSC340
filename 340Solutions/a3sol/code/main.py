# basics
import argparse
from optimizers import OptimizerGradientDescent, OptimizerGradientDescentLineSearch
from fun_obj import FunObjLeastSquares, FunObjRobustRegression
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# our code
import linear_models
import utils

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "2":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_models.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "2.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        """YOUR CODE FOR Q2.1"""
        n, d = X.shape
        v = np.ones(n)
        v[400:] = 0.1
        model = linear_models.WeightedLeastSquares()
        model.fit(X,y,v)
        utils.test_and_plot(model,X,y,title="Weighted Least Squares",filename="weighted_least_squares_outliers.pdf")


    elif question == "2.4":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        fun_obj = FunObjLeastSquares()
        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=100, verbose=False)
        model = linear_models.LinearModelGradientDescent(fun_obj, optimizer)
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")


    elif question == "2.4.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        """YOUR CODE HERE FOR Q2.4.1"""
        # TODO: Finish FunObjRobustRegression in fun_obj.py.
        fun_obj = FunObjRobustRegression()
        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=100, verbose=False)
        model = linear_models.LinearModelGradientDescent(fun_obj, optimizer)
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust_gd.pdf")

    elif question == "2.4.2":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        """YOUR CODE HERE FOR Q2.4.2"""
        # Produce the learning curves with
        # 1. OptimizerGradientDescent
        # 2. OptimizerGradientDescentLineSearch
        
        fun_obj = FunObjRobustRegression()
        optimizer = OptimizerGradientDescent(fun_obj, X, y, max_evals=100, verbose=False)
        model = linear_models.LinearModelGradientDescent(fun_obj, optimizer)
        fs1 = model.fit(X,y)

        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=100, verbose=False)
        model = linear_models.LinearModelGradientDescent(fun_obj, optimizer)
        fs2 = model.fit(X,y)

        plt.figure()
        plt.title("Objective function with gradient descent iterations")
        plt.ylabel("f")
        plt.xlabel("Gradient descent iteration")
        plt.plot(fs1, label="Gradient descent")
        plt.plot(fs2, label="Gradient descent with line search")
        plt.legend()
        plt.savefig("../figs/learning_curves_robust_regression.pdf")

    elif question == "3":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        X_test = data['Xtest']
        y_test = data['ytest']

        # Fit least-squares model
        model = linear_models.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,X_test,y_test,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "3.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        X_test = data['Xtest']
        y_test = data['ytest']

        """YOUR CODE HERE FOR Q3.1"""
        model = linear_models.LeastSquaresBias()
        model.fit(X, y)

        utils.test_and_plot(model,X,y,X_test,y_test,title="Least Squares, yes bias",filename="least_squares_yes_bias.pdf")

    elif question == "3.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        X_test = data['Xtest']
        y_test = data['ytest']

        err_trains = np.zeros(101)
        err_tests = np.zeros(101)
        for p in range(101):
            print("p=%d" % p)

            """YOUR CODE HERE FOR Q3.2"""
            model = linear_models.LeastSquaresPoly(p)
            model.fit(X, y)
            y_hat = model.predict(X)
            err_train = np.mean((y_hat - y) ** 2)
            err_trains[p] = err_train

            y_hat = model.predict(X_test)
            err_test = np.mean((y_hat - y_test) ** 2)
            err_tests[p] = err_test
        
        plt.figure()
        plt.plot(err_trains, label="training error")
        plt.plot(err_tests, label="test error")
        plt.xlabel("Degree of polynomial")
        plt.ylabel("Error")
        plt.ylim([0, 10000])
        plt.legend()
        plt.savefig("../figs/polynomial_error_curves.png")


    else:
        print("Unknown question: %s" % question)

