import os
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.optimize import approx_fprime

DATA_DIR = Path(__file__).parent.parent / "data"


def load_dataset(dataset_name, standardize=True, add_bias=True):
    with open((DATA_DIR / dataset_name).with_suffix(".pkl"), "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"].astype(np.int32)
    Xvalid = data["Xvalidate"]
    yvalid = data["yvalidate"].astype(np.int32)

    if standardize:
        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

    if add_bias:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

    return {"X": X, "y": y, "Xvalid": Xvalid, "yvalid": yvalid}


def standardize_cols(X, mu=None, sigma=None):
    "Standardize each column to have mean 0 and variance 1"
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.0

    return (X - mu) / sigma, mu, sigma


def check_gradient(model, X, y):
    # This checks that the gradient implementation is correct
    w = np.random.rand(*model.w.shape)
    f, g = model.fun_obj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(
        w, lambda w: model.fun_obj(w, X, y)[0], epsilon=1e-6
    )

    implemented_gradient = model.fun_obj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
        raise Exception(
            "User and numerical derivatives differ:\n%s\n%s"
            % (estimated_gradient[:5], implemented_gradient[:5])
        )
    else:
        print("User and numerical derivatives agree.")


def classification_error(y, yhat):
    return np.mean(y != yhat)


def ensure_1d(x):
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        return x.squeeze(axis=1)
    elif x.ndim == 0:
        return x[np.newaxis]
    else:
        raise ValueError(f"invalid shape {x.shape} for ensure_1d")
