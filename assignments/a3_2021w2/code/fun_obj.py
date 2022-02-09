import numpy as np
from scipy.optimize.optimize import approx_fprime

from utils import ensure_1d

"""
Implementation of function objects.
Function objects encapsulate the behaviour of an objective function that we optimize.
Simply put, implement evaluate(w, X, y) to get the numerical values corresponding to:
f, the function value (scalar) and
g, the gradient (vector).

Function objects are used with optimizers to navigate the parameter space and
to find the optimal parameters (vector). See optimizers.py.
"""


class FunObj:
    """
    Function object for encapsulating evaluations of functions and gradients
    """

    def evaluate(self, w, X, y):
        """
        Evaluates the function AND its gradient w.r.t. w.
        Returns the numerical values based on the input.
        """
        pass # this is because it's a base class, it will be implemented below.

    def check_correctness(self, w, X, y):
        n, d = X.shape
        w = ensure_1d(w)
        y = ensure_1d(y)
        estimated_gradient = approx_fprime(
            w,
            lambda w: self.evaluate(w, X, y)[0],
            epsilon=1e-6,
        )
        implemented_gradient = self.evaluate(w, X, y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print(
                "User and numerical derivatives differ: %s vs. %s"
                % (estimated_gradient, implemented_gradient)
            )
        else:
            print("User and numerical derivatives agree.")


class FunObjLeastSquares(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of least squares objective.
        Least squares objective is the sum of squared residuals.
        """
        # help avoid mistakes (as described in the assignment) by
        # potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        # Prediction is linear combination
        y_hat = X @ w
        # Residual is difference between ground truth and prediction
        # ("what's left" after your predicition)
        residuals = y - y_hat
        # Squared residuals gives us the objective function value
        f = 0.5 * np.sum(residuals ** 2)

        # Analytical gradient, written in mathematical form first
        # and then translated into Python.
        # The parentheses on the first term are just a small optimization:
        # this way, we do two matrix-vector multipliciations,
        # instead of a (more expensive) matrix-matrix mult and a matrix-vector
        g = X.T @ (X @ w) - X.T @ y
        return f, g


class FunObjRobustRegression(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of ROBUST least squares objective.
        """
        # help avoid mistakes (as described in the assignment) by
        # potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        """YOUR CODE HERE FOR Q2.3"""
        raise NotImplementedError()
