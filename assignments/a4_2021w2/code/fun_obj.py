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
        IMPORTANT: w is assumed to be a 1d-array, hence shaping will have to be handled.
        """
        raise NotImplementedError("This is a base class, don't call this")

    def check_correctness(self, w, X, y):
        n, d = X.shape
        estimated_gradient = approx_fprime(
            w, lambda w: self.evaluate(w, X, y)[0], epsilon=1e-6
        )
        _, implemented_gradient = self.evaluate(w, X, y)
        difference = estimated_gradient - implemented_gradient
        if np.max(np.abs(difference) > 1e-4):
            print(
                "User and numerical derivatives differ: %s vs. %s"
                % (estimated_gradient, implemented_gradient)
            )
        else:
            print("User and numerical derivatives agree.")


class LeastSquaresLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of least squares objective.
        Least squares objective is the sum of squared residuals.
        """
        # help avoid mistakes by potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        y_hat = X @ w
        m_residuals = y_hat - y  # minus residuals, slightly more convenient here

        # Loss is sum of squared residuals
        f = 0.5 * np.sum(m_residuals ** 2)

        # The gradient, derived mathematically then implemented here
        g = X.T @ m_residuals  # X^T X w - X^T y

        return f, g


class RobustRegressionLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of ROBUST least squares objective.
        """
        # help avoid mistakes by potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        y_hat = X @ w
        residuals = y - y_hat
        exp_residuals = np.exp(residuals)
        exp_minuses = np.exp(-residuals)

        f = np.sum(np.log(exp_minuses + exp_residuals))

        # s is the negative of the "soft sign"
        s = (exp_minuses - exp_residuals) / (exp_minuses + exp_residuals)
        g = X.T @ s

        return f, g


class LogisticRegressionLoss(FunObj):
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of logistics regression objective.
        """
        # help avoid mistakes by potentially reshaping our arguments
        w = ensure_1d(w)
        y = ensure_1d(y)

        Xw = X @ w
        yXw = y * Xw  # element-wise multiply; the y_i are in {-1, 1}

        # Calculate the function value
        f = np.sum(np.log(1 + np.exp(-yXw)))

        # Calculate the gradient value
        s = -y / (1 + np.exp(yXw))
        g = X.T @ s

        return f, g


class LogisticRegressionLossL2(LogisticRegressionLoss):
    def __init__(self, lammy):
        super().__init__()
        self.lammy = lammy

    def evaluate(self, w, X, y):
        w = ensure_1d(w)
        y = ensure_1d(y)

        """YOUR CODE HERE FOR Q2.1"""
        raise NotImplementedError()


class LogisticRegressionLossL0(FunObj):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        """
        Evaluates the function value of of L0-regularized logistics regression objective.
        """
        w = ensure_1d(w)
        y = ensure_1d(y)

        Xw = X @ w
        yXw = y * Xw  # element-wise multiply

        # Calculate the function value
        f = np.sum(np.log(1.0 + np.exp(-yXw))) + self.lammy * np.sum(w != 0)

        # We cannot differentiate the "length" function
        g = None
        return f, g


class SoftmaxLoss(FunObj):
    def evaluate(self, w, X, y):
        w = ensure_1d(w)
        y = ensure_1d(y)

        n, d = X.shape
        k = len(np.unique(y))

        """YOUR CODE HERE FOR Q3.4"""
        # Hint: you may want to use NumPy's reshape() or flatten()
        # to be consistent with our matrix notation.
        raise NotImplementedError()
