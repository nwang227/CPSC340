import numpy as np
from scipy.optimize.optimize import approx_fprime
from scipy.special import logsumexp

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
        Least squares objective is half the sum of squared residuals.
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


class LeastSquaresLossL2(LeastSquaresLoss):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        f_base, g_base = super().evaluate(w, X, y)
        f = f_base + self.lammy / 2 * (w @ w)
        g = g_base + self.lammy * w
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
        # logaddexp(a, b) = log(exp(a) + exp(b)), but more numerically stable
        f = np.logaddexp(0, -yXw).sum()

        # Calculate the gradient value
        with np.errstate(over="ignore"):  # overflowing here is okay: we get 0
            g_bits = -y / (1 + np.exp(yXw))
        g = X.T @ g_bits

        # 1 / (1 + exp(yXw)) = exp(-yXw) / (1 + exp(-yXw))

        # X.T @ (-y / (1 + exp(y * X w)))
        #

        return f, g


class LogisticRegressionLossL2(LogisticRegressionLoss):
    def __init__(self, lammy):
        super().__init__()
        self.lammy = lammy

    def evaluate(self, w, X, y):
        w = ensure_1d(w)
        y = ensure_1d(y)

        f_base, g_base = super().evaluate(w, X, y)
        f = f_base + (self.lammy / 2) * (w @ w)
        g = g_base + self.lammy * w
        return f, g


class KernelLogisticRegressionLossL2(FunObj):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, u, K, y):
        """
        Here u is the length-n vector defining our linear combination in
        the (potentially infinite-dimensional) Z space,
        and K is the Gram matrix K[i, i'] = k(x_i, x_{i'}).

        Note the L2 regularizer is in the transformed space too, not on u.
        """
        u = ensure_1d(u)
        y = ensure_1d(y)

        yKu = y * (K @ u)

        f = np.logaddexp(0, -yKu).sum() + (self.lammy / 2) * u @ K @ u

        with np.errstate(over="ignore"):  # overflowing here is okay: we get 0
            g_bits = -y / (1 + np.exp(yKu))
        g = K @ g_bits + self.lammy * K @ u

        return f, g


class LogisticRegressionLossL0(FunObj):
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        """
        Evaluates the function value of of L0-regularized logistics regression
        objective.
        """
        w = ensure_1d(w)
        y = ensure_1d(y)

        Xw = X @ w
        yXw = y * Xw  # element-wise multiply; the y should be in {-1, 1}

        # Calculate the function value
        f = np.logaddexp(0, -yXw).sum() + self.lammy * np.sum(w != 0)

        # We cannot differentiate the "length" function
        g = None
        return f, g


class SoftmaxLoss(FunObj):
    def evaluate(self, w, X, y):
        w = ensure_1d(w)
        y = ensure_1d(y)

        n, d = X.shape
        k = len(np.unique(y))

        W = w.reshape(k, d)

        # n by k; XW[i, c] is the dot product between example i and class c weights
        XW = X @ W.T

        # n by k: logsumexp_XW[i] = log( sum_c exp(x_i^T w_c) )
        logsumexp_XW = logsumexp(XW, axis=1)

        # n by k; p[i, c] = p(y_i = c | W, x_i)
        # we compute this "in log space" for numerical stability:
        # we never do an exp() of something positive,
        # and the only very-negative things we exp() are the ones that don't matter
        p = np.exp(XW - logsumexp_XW[:, np.newaxis])

        # n by k; one_hots[i, c] is 1 if y_i = c, 0 otherwise (appears in gradient)
        one_hots = np.eye(k)[y]

        # now to compute the function value:
        Xw_of_y = XW[np.arange(n), y]  # n; confusingly, : would NOT work in first index
        f = -Xw_of_y.sum() + logsumexp_XW.sum()  # scalar

        # n by k by d; G_terms[i, c, j] = x_ij [p(y_i=c | W, x_i) - 1(y_i = c)]
        G_terms = X[:, np.newaxis, :] * (p - one_hots)[:, :, np.newaxis]
        G = G_terms.sum(axis=0)  # k by d
        g = G.reshape(-1)  # kd

        return f, g
