from kernels import KernelGaussianRBF, KernelLinear, KernelPolynomial
from linear_models import LinearModelGradientDescent, LogRegClassifier, LogRegClassifierKernel
from optimizers import OptimizerGradientDescent, OptimizerGradientDescentLineSearch, OptimizerStochasticGradient
from fun_obj import FunObjLeastSquares, FunObjLeastSquaresL2, FunObjLogRegL2, FunObjLogRegL2Kernel
from learning_rate_getters import LearningRateGetterConstant, LearningRateGetterInverse, LearningRateGetterInverseSqrt, LearningRateGetterInverseSquared
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import utils
from compressors import PCA

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        data =  pickle.load(f)

    if filename == "nonLinearData.pkl":
        X, y = data['X'], data['y']
        X_train, X_valid, y_train, y_valid = train_test_split(X,y,random_state=0)

        return X, y, X_train, y_train, X_valid, y_valid

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    if question == "1":
        X_train, y_train, X_train, y_train, X_valid, y_valid = load_dataset('nonLinearData.pkl')

        # Standard logistic regression
        # NOTE: For A5, I refactored the optimizer constructor to exclude fun_obj argument.
        # See optimizers.py documentation for more detail.
        fun_obj = FunObjLogRegL2(1)
        optimizer = OptimizerGradientDescentLineSearch()
        model_lr = LogRegClassifier(fun_obj, optimizer)
        model_lr.fit(X_train, y_train)

        print("Training error {:.3f}".format(np.mean(model_lr.predict(X_train) != y_train)))
        print("Validation error {:.3f}".format(np.mean(model_lr.predict(X_valid) != y_valid)))

        utils.plot_classifier(model_lr, X_train, y_train)
        utils.savefig("logReg.png")
        
        # kernel logistic regression with a linear kernel
        fun_obj = FunObjLogRegL2(1)
        optimizer = OptimizerGradientDescentLineSearch()
        kernel = KernelLinear()
        model_lr_kernel = LogRegClassifierKernel(fun_obj, optimizer, kernel)
        model_lr_kernel.fit(X_train, y_train)

        print("Training error {:.3f}".format(np.mean(model_lr_kernel.predict(X_train) != y_train)))
        print("Validation error {:.3f}".format(np.mean(model_lr_kernel.predict(X_valid) != y_valid)))

        utils.plot_classifier(model_lr_kernel, X_train, y_train)
        utils.savefig("logRegLinearKernel.png")

    elif question == "1.1":
        X_train, y_train, X_train, y_train, X_valid, y_valid = load_dataset('nonLinearData.pkl')

        """YOUR CODE HERE FOR Q1.1"""
        fun_obj = FunObjLogRegL2(0.01)
        optimizer = OptimizerGradientDescentLineSearch()
        kernel = KernelPolynomial(2)
        model = LogRegClassifierKernel(fun_obj, optimizer, kernel)
        model.fit(X_train, y_train)

        print("Training error {:.3f}".format(np.mean(model.predict(X_train) != y_train)))
        print("Validation error {:.3f}".format(np.mean(model.predict(X_valid) != y_valid)))

        utils.plot_classifier(model, X_train, y_train)
        utils.savefig("logRegPoly.png")

        fun_obj = FunObjLogRegL2(0.01)
        optimizer = OptimizerGradientDescentLineSearch()
        kernel = KernelGaussianRBF(0.5)
        model = LogRegClassifierKernel(fun_obj, optimizer, kernel)
        model.fit(X_train, y_train)

        print("Training error {:.3f}".format(np.mean(model.predict(X_train) != y_train)))
        print("Validation error {:.3f}".format(np.mean(model.predict(X_valid) != y_valid)))

        utils.plot_classifier(model, X_train, y_train)
        utils.savefig("logRegRBF.png")


    elif question == "1.2":
        X_train, y_train, X_train, y_train, X_valid, y_valid = load_dataset('nonLinearData.pkl')
        
        """YOUR CODE HERE FOR Q1.2"""
        sigmas = 10. ** np.array([-2, -1, 0, 1, 2])
        lammys = 10. ** np.array([-4, -3, -2, -1, 0])

        best_train_sigma = None
        best_train_lammy = None
        best_valid_sigma = None
        best_valid_lammy = None
        best_err_valid = np.inf
        best_err_train = np.inf
        for sigma in sigmas:
            for lammy in lammys:
                fun_obj = FunObjLogRegL2Kernel(lammy)
                optimizer = OptimizerGradientDescentLineSearch()
                kernel = KernelGaussianRBF(sigma)
                model = LogRegClassifierKernel(fun_obj, optimizer, kernel)
                model.fit(X_train, y_train)

                y_hat = model.predict(X_train)
                err_train = np.mean(y_hat != y_train)
                
                if err_train < best_err_train:
                    best_err_train = err_train
                    best_train_sigma = sigma
                    best_train_lammy = lammy

                y_hat = model.predict(X_valid)
                err_valid = np.mean(y_hat != y_valid)
                
                if err_valid < best_err_valid:
                    best_err_valid = err_valid
                    best_valid_sigma = sigma
                    best_valid_lammy = lammy

        print("best training error: {:.5f}".format(best_err_train))
        print("lammy for best training error: {:.5f}".format(best_train_lammy))
        print("sigma for best training error: {:.5f}".format(best_train_sigma))

        fun_obj = FunObjLogRegL2Kernel(best_train_lammy)
        optimizer = OptimizerGradientDescentLineSearch()
        kernel = KernelGaussianRBF(best_train_sigma)
        model = LogRegClassifierKernel(fun_obj, optimizer, kernel)
        model.fit(X_train, y_train)
        utils.plot_classifier(model, X_train, y_train)
        utils.savefig("logRegRBF_best_train.png")

        print("best validation error: {:.5f}".format(best_err_valid))
        print("lammy for best validation error: {:.5f}".format(best_valid_lammy))
        print("sigma for best validation error: {:.5f}".format(best_valid_sigma))

        fun_obj = FunObjLogRegL2Kernel(best_valid_lammy)
        optimizer = OptimizerGradientDescentLineSearch()
        kernel = KernelGaussianRBF(best_valid_sigma)
        model = LogRegClassifierKernel(fun_obj, optimizer, kernel)
        model.fit(X_train, y_train)
        utils.plot_classifier(model, X_train, y_train)
        utils.savefig("logRegRBF_best_valid.png")


    elif question == "3.2":
        data = load_dataset('animals.pkl')
        X_train = data['X']
        animal_names = data['animals']
        trait_names = data['traits']

        # Standardize features
        X_train_standardized, mu, sigma = utils.standardize_cols(X_train)
        n, d = X_train_standardized.shape

        # Matrix plot
        plt.figure()
        plt.imshow(X_train_standardized)
        utils.savefig("animals_matrix.png")

        # 2D visualization
        np.random.seed(1234)  # set seed to be consistent with skeleton code
        j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
        random_is = np.random.choice(n, 20, replace=False)  # choose 10 random examples

        plt.figure()
        plt.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
        for i in random_is:
            xy = X_train_standardized[i, [j1, j2]]
            plt.annotate(animal_names[i], xy=xy)
        utils.savefig("animals_random.png")
        
        """YOUR CODE HERE FOR Q3.2"""

        model = PCA(2)
        model.fit(X_train)
        Z = model.compress(X_train)

        plt.scatter(Z[:, 0], Z[:, 1])
        for i in random_is:
            xy = Z[i, [0, 1]]
            plt.annotate(animal_names[i], xy=xy)
        utils.savefig("animals_pca.png")

        # For Q3.2.2 and Q3.2.3
        print(trait_names[np.argmax(np.abs(model.W[0, :]))])
        print(trait_names[np.argmax(np.abs(model.W[1, :]))])

        # For Q3.3
        X_centered = X_train - model.mu
        variance_explained = 1 - (np.sum((Z@model.W - X_centered) ** 2) / np.sum(X_centered ** 2))
        print("Variance explained: {:.3f}".format(variance_explained))

    elif question == "4":
        # Load dynamics dataset
        data = load_dataset("dynamics.pkl")
        X_train = data['X_train']
        y_train = data['y_train']
        X_valid = data['X_valid']
        y_valid = data['y_valid']
        X_train_standardized, mu, sigma = utils.standardize_cols(X_train)
        X_valid_standardized, _, _ = utils.standardize_cols(X_valid, mu, sigma)

        # Train ordinary regularized least squares
        fun_obj = FunObjLeastSquares()
        optimizer = OptimizerGradientDescentLineSearch()
        model = LinearModelGradientDescent(fun_obj, optimizer, check_correctness_yes=False)
        model.fit(X_train_standardized, y_train)
        print(model.fs)  # ~700 seems to be the global minimum.

        y_hat = model.predict(X_train_standardized)
        print("Training error: {:.3f}".format(np.mean((y_hat - y_train) ** 2)))

        y_hat = model.predict(X_valid_standardized)
        print("Validation error: {:.3f}".format(np.mean((y_hat - y_valid) ** 2)))

        # Plot the learning curve!
        plt.figure()
        plt.plot(model.fs)
        plt.xlabel("Gradient descent iterations")
        plt.ylabel("Objective function f value")
        utils.savefig("gd_line_search_curve.png")

    elif question == "4.1":
        # Load dynamics dataset
        data = load_dataset("dynamics.pkl")
        X_train = data['X_train']
        y_train = data['y_train']
        X_valid = data['X_valid']
        y_valid = data['y_valid']
        X_train_standardized, mu, sigma = utils.standardize_cols(X_train)
        X_valid_standardized, _, _ = utils.standardize_cols(X_valid, mu, sigma)

        """YOUR CODE HERE FOR Q4.1"""
        batch_sizes = [1, 10, 100]
        # fun_obj = FunObjLeastSquaresL2(1)
        fun_obj = FunObjLeastSquares()
        child_optimizer = OptimizerGradientDescent()
        learning_rate_getter = LearningRateGetterConstant(3e-4)
        for batch_size in batch_sizes:
            optimizer = OptimizerStochasticGradient(child_optimizer, learning_rate_getter, batch_size, max_evals=10)
            model = LinearModelGradientDescent(fun_obj, optimizer)
            model.fit(X_train_standardized, y_train)
            
            err_train = np.mean((model.predict(X_train_standardized) - y_train) ** 2)
            err_valid = np.mean((model.predict(X_valid_standardized) - y_valid) ** 2)

            print("Batch size: {:d}\tTraining error: {:.3f}\tValidation error: {:.3f}".format(batch_size, err_train, err_valid))

    elif question == "4.3":
        # Load dynamics dataset
        data = load_dataset("dynamics.pkl")
        X_train = data['X_train']
        y_train = data['y_train']
        X_valid = data['X_valid']
        y_valid = data['y_valid']
        X_train_standardized, mu, sigma = utils.standardize_cols(X_train)
        X_valid_standardized, _, _ = utils.standardize_cols(X_valid, mu, sigma)

        """YOUR CODE HERE FOR Q4.3"""

        learning_rate_getters = [
            LearningRateGetterConstant(3e-1),
            LearningRateGetterInverse(3e-1),
            LearningRateGetterInverseSquared(3e-1),
            LearningRateGetterInverseSqrt(3e-1)
        ]
        plot_labels = [
            "constant",
            "inverse",
            "inverse_squared",
            "inverse_sqrt"
        ]
        plt.figure()
        for i, learning_rate_getter in enumerate(learning_rate_getters):
            fun_obj = FunObjLeastSquares()
            child_optimizer = OptimizerGradientDescent()
            optimizer = OptimizerStochasticGradient(child_optimizer, learning_rate_getter, 10, max_evals=50)
            model = LinearModelGradientDescent(fun_obj, optimizer)
            model.fit(X_train_standardized, y_train)
            err_train = np.mean((model.predict(X_train) - y_train) ** 2)
            err_valid = np.mean((model.predict(X_valid) - y_valid) ** 2)

            print("Learning rate: {:s}\tTraining error: {:.3f}\tValidation error: {:.3f}".format(plot_labels[i], err_train, err_valid))
            plt.plot(model.fs, label=plot_labels[i])
        
        plt.axhline(700, label="line_search", linestyle="--")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Objective function f value")
        utils.savefig("sgd_learning_curves.png")

    else:
        print("Unknown question: {:s}".format(question))
