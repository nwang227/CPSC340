#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import pickle
import gzip
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelBinarizer


# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from encoders import (
    LinearEncoderGradient,
    PCAEncoder,
    NonLinearEncoderMultiLayer,
)
from linear_models import LinearModelMultiOutput, MulticlassLinearClassifier
from learning_rate_getters import ConstantLR
from fun_obj import (
    MLPLogisticRegressionLossL2,
    PCAFactorsLoss,
    PCAFeaturesLoss,
    RobustPCAFactorsLoss,
    RobustPCAFeaturesLoss,
    SoftmaxLoss,
    CollaborativeFilteringWLoss,
    CollaborativeFilteringZLoss,
)
from neural_net import NeuralNet
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    StochasticGradient,
)
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
    parser.add_argument(
        "-q", "--question", required=True, choices=sorted(_funcs.keys()) + ["all"]
    )
    args = parser.parse_args()

    utils.welcome()  # it is permissible to comment out this line

    args = parser.parse_args()
    if args.question == "all":
        for q in sorted(_funcs.keys()):
            start = f"== {q} "
            print("\n" + start + "=" * (80 - len(start)))
            run(q)
    else:
        return run(args.question)


@handle("1")
def q1():
    X_train = load_dataset("highway.pkl")["X"].astype(float) / 255.0
    n, d = X_train.shape
    h, w = 64, 64  # height and width of each image
    k = 5  # number of PCs
    threshold = 0.1  # threshold for being considered "foreground"

    # PCA with SVD
    model = PCAEncoder(k)
    model.fit(X_train)
    Z = model.encode(X_train)
    X_hat = model.decode(Z)

    # PCA with alternating minimization
    fun_obj_w = PCAFactorsLoss()
    fun_obj_z = PCAFeaturesLoss()
    optimizer_w = GradientDescentLineSearch(max_evals=100, verbose=False)
    optimizer_z = GradientDescentLineSearch(max_evals=100, verbose=False)
    model = LinearEncoderGradient(k, fun_obj_w, fun_obj_z, optimizer_w, optimizer_z)
    model.fit(X_train)
    Z_alt = model.encode(X_train)
    X_hat_alt = model.decode(Z_alt)

    for i in range(10):
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].set_title("$X$")
        ax[0, 0].imshow(X_train[i].reshape(h, w).T, cmap="gray")

        ax[0, 1].set_title(r"$\hat{X}$ (L2)")
        ax[0, 1].imshow(X_hat[i].reshape(h, w).T, cmap="gray")

        ax[0, 2].set_title(r"$|x_i-\hat{x_i}|$>threshold (L2)")
        ax[0, 2].imshow(
            (np.abs(X_train[i] - X_hat[i]) < threshold).reshape(h, w).T, cmap="gray"
        )

        ax[1, 0].set_title("$X$")
        ax[1, 0].imshow(X_train[i].reshape(h, w).T, cmap="gray")

        ax[1, 1].set_title(r"$\hat{X}$ (L1)")
        ax[1, 1].imshow(X_hat_alt[i].reshape(h, w).T, cmap="gray")

        ax[1, 2].set_title(r"$|x_i-\hat{x_i}|$>threshold (L1)")
        ax[1, 2].imshow(
            (np.abs(X_train[i] - X_hat_alt[i]) < threshold).reshape(h, w).T, cmap="gray"
        )

        utils.savefig(f"pca_highway_{i:03}.jpg", fig=fig)
        plt.close(fig)


@handle("1.3")
def q1_3():
    X_train = load_dataset("highway.pkl")["X"].astype(float) / 255.0
    n, d = X_train.shape
    h, w = 64, 64  # height and width of each image
    k = 5  # number of PCs
    threshold = 0.1  # threshold for being considered "foreground"

    # PCA with SVD
    model = PCAEncoder(k)
    model.fit(X_train)
    Z = model.encode(X_train)
    X_hat = model.decode(Z)

    # TODO: Implement function objects for robust PCA in fun_obj.py
    fun_obj_w = RobustPCAFactorsLoss(1e-6)
    fun_obj_z = RobustPCAFeaturesLoss(1e-6)
    optimizer_w = GradientDescentLineSearch(max_evals=100, verbose=False)
    optimizer_z = GradientDescentLineSearch(max_evals=100, verbose=False)
    model = LinearEncoderGradient(k, fun_obj_w, fun_obj_z, optimizer_w, optimizer_z)
    model.fit(X_train)
    Z_alt = model.encode(X_train)
    X_hat_alt = model.decode(Z_alt)

    for i in range(10):
        fig, ax = plt.subplots(2, 3)
        ax[0, 0].set_title("$X$")
        ax[0, 0].imshow(X_train[i].reshape(h, w).T, cmap="gray")

        ax[0, 1].set_title(r"$\hat{X}$ (L2)")
        ax[0, 1].imshow(X_hat[i].reshape(h, w).T, cmap="gray")

        ax[0, 2].set_title(r"$|x_i-\hat{x_i}|$>threshold (L2)")
        ax[0, 2].imshow(
            (np.abs(X_train[i] - X_hat[i]) < threshold).reshape(h, w).T, cmap="gray"
        )

        ax[1, 0].set_title("$X$")
        ax[1, 0].imshow(X_train[i].reshape(h, w).T, cmap="gray")

        ax[1, 1].set_title(r"$\hat{X}$ (L1)")
        ax[1, 1].imshow(X_hat_alt[i].reshape(h, w).T, cmap="gray")

        ax[1, 2].set_title(r"$|x_i-\hat{x_i}|$>threshold (L1)")
        ax[1, 2].imshow(
            (np.abs(X_train[i] - X_hat_alt[i]) < threshold).reshape(h, w).T, cmap="gray"
        )

        utils.savefig(f"robustpca_highway_{i:03}.jpg", fig)
        plt.close(fig)

q1_3()

@handle("2")
def q2():
    ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("../data/ml-latest-small/movies.csv", index_col=0)

    print("Sample of the ratings dataframe:")
    print(ratings.head())
    print("\nSample of the movies dataframe:")
    print(movies.head())

    n = len(set(ratings["userId"]))
    d = len(set(ratings["movieId"]))
    print("Number of users:", n)
    print("Number of movies:", d)

    print("Number of ratings:", len(ratings))

    Y_train, Y_valid = utils.create_rating_matrix(ratings, n, d, "userId", "movieId")

    # get the average rating in Y_train
    print("The average rating in the training set: %.2f" % np.nanmean(Y_train))
    # equivalent to the previous line, but implemented differently
    print(
        "The average rating in the training set (again): %.2f"
        % np.mean(Y_train[~np.isnan(Y_train)])
    )


@handle("2.1")
def q2_1():
    ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("../data/ml-latest-small/movies.csv", index_col=0)

    n = len(set(ratings["userId"]))
    d = len(set(ratings["movieId"]))
    print("Number of users:", n)
    print("Number of movies:", d)

    print("Number of ratings:", len(ratings))

    Y_train, Y_valid = utils.create_rating_matrix(ratings, n, d, "userId", "movieId")

    raise NotImplementedError()



@handle("2.2")
def q2_2():
    ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("../data/ml-latest-small/movies.csv", index_col=0)

    n = len(set(ratings["userId"]))
    d = len(set(ratings["movieId"]))

    # don't make it a sparse matrix for now, for simplicity, despite the inefficiency
    Y_train, Y_valid = utils.create_rating_matrix(ratings, n, d, "userId", "movieId")

    avg_rating = np.nanmean(Y_train)

    # PCA with alternating minimization
    k = 50
    # Below: you need to use the same lammyZ in both cases, and the same lammyW in both cases
    # TODO next year: improve this code
    fun_obj_w = CollaborativeFilteringWLoss(lammyZ=1, lammyW=1)
    fun_obj_z = CollaborativeFilteringZLoss(lammyZ=1, lammyW=1)

    # smaller version for checking the gradient, otherwise it's slow
    k_check = 3
    n_check = 100
    d_check = 50
    Y_train_check = Y_train[:n_check, :d_check]
    fun_obj_w.check_correctness(
        np.random.rand(k_check * d_check),
        np.random.rand(n_check, k_check),
        Y_train_check,
        epsilon=1e-6,
    )
    fun_obj_z.check_correctness(
        np.random.rand(n_check * k_check),
        np.random.rand(k_check, d_check),
        Y_train_check,
        epsilon=1e-6,
    )

    optimizer_w = GradientDescentLineSearch(max_evals=100, verbose=False)
    optimizer_z = GradientDescentLineSearch(max_evals=100, verbose=False)
    model = LinearEncoderGradient(
        k, fun_obj_w, fun_obj_z, optimizer_w, optimizer_z, centering="all"
    )
    # centering="all" means take the mean over all non-NaN elements of Y,
    # not the mean per each column
    model.fit(Y_train)

    Y_hat = model.Z @ model.W + model.mu

    RMSE_train = np.sqrt(np.nanmean((Y_hat - Y_train) ** 2))
    print("Train RMSE of ratings: %.2f" % RMSE_train)

    RMSE_valid = np.sqrt(np.nanmean((Y_hat - Y_valid) ** 2))
    print("Valid RMSE of ratings: %.2f" % RMSE_valid)


@handle("2.3")
def q2_3():
    ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("../data/ml-latest-small/movies.csv", index_col=0)

    n = len(set(ratings["userId"]))
    d = len(set(ratings["movieId"]))

    avg_rating = np.mean(ratings["rating"])
    # don't make it a sparse matrix for now, for simplicity, despite the inefficiency
    Y_train, Y_valid = utils.create_rating_matrix(ratings, n, d, "userId", "movieId")

    avg_rating = np.nanmean(Y_train)

    k = 50
    fun_obj_w = CollaborativeFilteringWLoss(lammyZ=1, lammyW=1)
    fun_obj_z = CollaborativeFilteringZLoss(lammyZ=1, lammyW=1)


    optimizer_w = GradientDescentLineSearch(max_evals=100, verbose=False)
    optimizer_z = GradientDescentLineSearch(max_evals=100, verbose=False)
    model = LinearEncoderGradient(
        k, fun_obj_w, fun_obj_z, optimizer_w, optimizer_z, centering="all"
    )
    # centering="all" means take the mean over all non-NaN elements of Y,
    # not the mean per each column
    model.fit(Y_train)

    Y_hat = model.Z @ model.W + model.mu

    print(
        "Train RMSE if you just guess the average: %.2f" % 
        np.sqrt(np.nanmean((avg_rating - Y_train) ** 2))
    )
    print(
        "Valid RMSE if you just guess the average: %.2f" %
        np.sqrt(np.nanmean((avg_rating - Y_valid) ** 2))
    )

    RMSE_train = np.sqrt(np.nanmean((Y_hat - Y_train) ** 2))
    print("Train RMSE of ratings: %.2f" % RMSE_train)

    RMSE_valid = np.sqrt(np.nanmean((Y_hat - Y_valid) ** 2))
    print("Valid RMSE of ratings: %.2f" % RMSE_valid)

    # print("Max abs of W:", np.max(np.abs(model.W)))
    # print("Max abs of Z:", np.max(np.abs(model.Z)))


@handle("2.4")
def q2_4():
    ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")
    movies = pd.read_csv("../data/ml-latest-small/movies.csv", index_col=0)

    n = len(set(ratings["userId"]))
    d = len(set(ratings["movieId"]))

    # don't make it a sparse matrix for now, for simplicity, despite the inefficiency
    Y_train, Y_valid = utils.create_rating_matrix(ratings, n, d, "userId", "movieId")
    avg_rating = np.nanmean(Y_train)

    raise NotImplementedError()



@handle("3")
def q3():
    # TODO: use scikit-learn's loader instead, it's better...
    with gzip.open(Path("..", "data", "mnist.pkl.gz"), "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    # Use these for softmax classifier
    X_train, y_train = train_set
    X_valid, y_valid = valid_set

    binarizer = LabelBinarizer()
    Y_train = binarizer.fit_transform(y_train)

    n, d = X_train.shape
    _, k = Y_train.shape  # k is the number of classes

    fun_obj = SoftmaxLoss()
    child_optimizer = GradientDescent()
    learning_rate_getter = ConstantLR(1e-3)
    optimizer = StochasticGradient(
        child_optimizer, learning_rate_getter, batch_size=500, max_evals=10
    )
    model = MulticlassLinearClassifier(fun_obj, optimizer)
    # model = SGDClassifier(alpha=0.001, max_iter=10)

    # t = time.time()
    model.fit(X_train, y_train)
    # print("Fitting took {:f} seconds".format((time.time() - t)))

    # Compute training error
    y_hat = model.predict(X_train)
    err_train = np.mean(y_hat != y_train)
    print("Training error = ", err_train)

    # Compute validation error
    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print("Validation error     = ", err_valid)


@handle("3.2")
def q3_2():
    with gzip.open(Path("..", "data", "mnist.pkl.gz"), "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    # Use these y-values for softmax classifier
    X_train, y_train = train_set
    X_valid, y_valid = valid_set

    # Use these for training our MLP classifier
    binarizer = LabelBinarizer()
    Y_train = binarizer.fit_transform(y_train)

    n, d = X_train.shape
    _, k = Y_train.shape  # k is the number of classes


    # Assemble a neural network
    # put hidden layer dimensions to increase the number of layers in encoder
    hidden_feature_dims = []
    output_dim = 2

    # First, initialize an encoder and a predictor
    layer_sizes = [d, *hidden_feature_dims, output_dim]
    encoder = NonLinearEncoderMultiLayer(layer_sizes)
    predictor = LinearModelMultiOutput(output_dim, k)

    # Function object will associate the encoder and the predictor during training
    fun_obj = MLPLogisticRegressionLossL2(encoder, predictor, 1.)

    # Choose optimization strategy
    child_optimizer = GradientDescent()
    learning_rate_getter = ConstantLR(1e-2)
    # learning_rate_getter = LearningRateGetterInverseSqrt(1e0)
    optimizer = StochasticGradient(
        child_optimizer, learning_rate_getter, 500, max_evals=10
    )

    # Assemble!
    model = NeuralNet(fun_obj, optimizer, encoder, predictor, classifier_yes=True)

    t = time.time()
    model.fit(X_train, Y_train)
    print("Fitting took {:f} seconds".format((time.time() - t)))

    # Compute training error
    y_hat = model.predict(X_train)
    err_train = np.mean(y_hat != y_train)
    print("Training error = ", err_train)

    # Compute validation error
    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print("Validation error     = ", err_valid)


@handle("3.3")
def q3_3():
    data = load_dataset("sinusoids.pkl")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_valid = data["X_valid"]
    y_valid = data["y_valid"]

    n, d = X_train.shape
    k = len(np.unique(y_train))

    Y_train = np.stack([1 - y_train, y_train], axis=1).astype(np.uint)

    fig, ax = plt.subplots()
    for c, color in [(0, "b"), (1, "r")]:
        in_c = y_train == c
        ax.scatter(X_train[in_c, 0], X_train[in_c, 1], color=color, label=f"class {c}")
    ax.set_title("Sinusoid data, non-convex but separable.")
    utils.savefig("sinusoids.png", fig)
    plt.close(fig)

    X_train_standardized, mu, sigma = utils.standardize_cols(X_train)
    X_valid_standardized, _, _ = utils.standardize_cols(X_valid, mu, sigma)

    for hidden_feature_dims in [[4], [2], [2, 2]]:
        # We're running this several times, for different encoder architectures.
        output_dim = 2
        layer_sizes = [d, *hidden_feature_dims, output_dim]

        title = (
            f"hidden dimensions={hidden_feature_dims}, output dimension={output_dim}"
        )
        fn_suffix = f"{hidden_feature_dims}_{output_dim}"
        print("\nRunning with " + title)

        # for reproducibility of the solution
        np.random.seed(10)

        best_err_valid = np.inf
        best_model = None
        for seed in range(20):  # "grid search over random seeds"
            # First, initialize an encoder and a predictor
            encoder = NonLinearEncoderMultiLayer(layer_sizes)
            predictor = LinearModelMultiOutput(output_dim, k)
            fun_obj = MLPLogisticRegressionLossL2(encoder, predictor, 0.0)
            optimizer = GradientDescentLineSearch()
            model = NeuralNet(
                fun_obj, optimizer, encoder, predictor, classifier_yes=True
            )
            for _ in range(10):
                # "continual warm-start with resets":
                # one brute-force method to fight NP-hard problems!
                # calling fit() will reset the optimizer state,
                # but the encoder and predictor's parameters will stay intact.
                model.fit(X_train_standardized, Y_train)

                # Comput training error
                y_hat = model.predict(X_train_standardized)
                err_train = np.mean(y_hat != y_train)

                # Compute validation error
                y_hat = model.predict(X_valid_standardized)
                err_valid = np.mean(y_hat != y_valid)

                if err_valid < best_err_valid:
                    best_err_valid = err_valid
                    best_model = model

                    print("Training error = ", err_train)
                    print("Validation error     = ", err_valid)

        # Visualize learned features
        Z_train, _ = best_model.encode(X_train_standardized)

        fig, ax = plt.subplots()
        for c, color in [(0, "b"), (1, "r")]:
            in_c = y_train == c
            ax.scatter(
                X_train[in_c, 0], X_train[in_c, 1], color=color, label=f"class {c}"
            )
        ax.set_xlabel("$z_{1}$")
        ax.set_ylabel("$z_{2}$")
        ax.set_title("Learned features of sinusoid data\n" + title)
        utils.savefig(f"sinusoids_learned_features_{fn_suffix}.png", fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        utils.plot_classifier(
            best_model.predictor, Z_train, y_train, need_argmax=True, ax=ax
        )
        ax.set_xlabel("$z_{1}$")
        ax.set_ylabel("$z_{2}$")
        ax.set_title("Decision boundary in transformed feature space\n" + title)
        utils.savefig(f"sinusoids_linear_boundary_{fn_suffix}.png", fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        utils.plot_classifier(best_model, X_train_standardized, y_train, ax=ax)
        ax.set_xlabel("$x_{1}$")
        ax.set_ylabel("$x_{2}$")
        ax.set_title("Decision boundary in original feature space\n" + title)
        utils.savefig(f"sinusoids_decision_boundary_{fn_suffix}.png", fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
