from pathlib import Path

import numpy as np
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix as sparse_matrix


def test_and_plot(model, X, y, Xtest=None, ytest=None, title=None, filename=None):

    # Compute training error
    yhat = model.predict(X)
    trainError = np.mean((yhat - y) ** 2)
    print("Training error = %.1f" % trainError)

    # Compute test error
    if Xtest is not None and ytest is not None:
        yhat = model.predict(Xtest)
        testError = np.mean((yhat - ytest) ** 2)
        print("Validation error     = %.1f" % testError)

    # Plot data and model
    plt.figure()
    plt.scatter(X, y, color="b")

    # Choose points to evaluate the function
    Xgrid = np.linspace(np.min(X), np.max(X), 1000)[:, None]
    ygrid = model.predict(Xgrid)
    plt.plot(Xgrid, ygrid, color="r", lw=2)

    if title is not None:
        plt.title(title)

    if filename is not None:
        filename = Path("..", "figs", filename)
        print("Saving to", filename)
        plt.savefig(filename)


def ensure_1d(x):
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        return x.squeeze(axis=1)
    elif x.ndim == 0:
        return x[np.newaxis]
    else:
        raise ValueError(f"invalid shape {x.shape} for ensure_1d")
