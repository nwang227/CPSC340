from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from sklearn.model_selection import train_test_split


def load_dataset(filename):
    with open(Path("..", "data", filename), "rb") as f:
        return pickle.load(f)


def load_trainval(filename):
    d = load_dataset(filename)
    return d["X_train"], d["y_train"], d["X_valid"], d["y_valid"]


def load_and_split(filename, **kwargs):
    data = load_dataset(filename)
    X = data["X"]
    y = data["y"]
    kwargs.setdefault("random_state", 0)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, **kwargs)

    return X_train, y_train, X_valid, y_valid


def savefig(fname, fig=None, verbose=True):
    path = Path("..", "figs", fname)
    (plt if fig is None else fig).savefig(path, bbox_inches="tight", pad_inches=0)
    if verbose:
        print(f"Figure saved as '{path}'")


def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.0

    return (X - mu) / sigma, mu, sigma


def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array
    """

    # add extra dimensions, so function still works for X and/or Xtest are 1-D arrays.
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if Xtest.ndim == 1:
        Xtest = Xtest[np.newaxis, :]

    return (
        np.sum(X ** 2, axis=1)[:, np.newaxis]
        + np.sum(Xtest ** 2, axis=1)[np.newaxis, :]
        - 2 * X @ Xtest.T
    )


def classification_error(y, yhat):
    return np.mean(y != yhat)


def check_gradient(model, X, y, dimensionality, verbose=True):
    # This checks that the gradient implementation is correct
    w = np.random.rand(dimensionality)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(
        w, lambda w: model.funObj(w, X, y)[0], epsilon=1e-6
    )

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-3):
        raise ValueError(
            "User and numerical derivatives differ:\n"
            f"{implemented_gradient[:5]}\n"
            f"{estimated_gradient[:5]}"
        )
    else:
        if verbose:
            print("User and numerical derivatives agree.")


def plot_classifier(model, X, y):
    """plots the decision boundary of the model and the scatterpoints
       of the target values 'y'.

    Assumptions
    -----------
    y : it should contain two classes: '1' and '2'

    Parameters
    ----------
    model : the trained model which has the predict function

    X : the N by D feature array

    y : the N element vector corresponding to the target values

    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1
    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1

    x1_line = np.linspace(x1_min, x1_max, 200)
    x2_line = np.linspace(x2_min, x2_max, 200)

    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)

    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model.predict(mesh_data)
    y_pred = np.reshape(y_pred, x1_mesh.shape)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([x1_mesh.min(), x1_mesh.max()])
    ax.set_ylim([x2_mesh.min(), x2_mesh.max()])

    ax.contourf(
        x1_mesh,
        x2_mesh,
        -y_pred.astype(int),  # unsigned int causes problems with negative sign... o_O
        cmap=plt.cm.RdBu,
        alpha=0.6,
    )

    y_vals = np.unique(y)
    ax.scatter(
        x1[y == y_vals[0]], x2[y == y_vals[0]], color="b", label="class %+d" % y_vals[0]
    )
    ax.scatter(
        x1[y == y_vals[1]], x2[y == y_vals[1]], color="r", label="class %+d" % y_vals[1]
    )
    ax.legend()
    return fig


def ensure_1d(x):
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        return x.squeeze(axis=1)
    elif x.ndim == 0:
        return x[np.newaxis]
    else:
        raise ValueError(f"invalid shape {x.shape} for ensure_1d")
