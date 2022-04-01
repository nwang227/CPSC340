import numpy as np
from scipy.optimize.optimize import approx_fprime

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
        raise NotImplementedError

    def check_correctness(self, w, X, y):
        n, d = X.shape
        estimated_gradient = approx_fprime(w, lambda w: self.evaluate(w, X, y)[0], epsilon=1e-6)
        _, implemented_gradient = self.evaluate(w, X, y)
        difference = estimated_gradient - implemented_gradient
        if np.max(np.abs(difference) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient))
        else:
            print('User and numerical derivatives agree.')

class FunObjLeastSquares(FunObj):
    
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of least squares objective.
        Least squares objective is the sum of squared residuals.
        """

        # Prediction is linear combination
        y_hat = X@w
        # Residual is difference between prediction and ground truth
        residuals = y_hat - y
        # Squared residuals gives us the objective function value
        f = 0.5 * np.sum(residuals ** 2)
        # Analytical gradient, written in mathematical form first
        # and then translated into Python
        g = X.T@X@w - X.T@y
        return f, g

class FunObjRobustRegression(FunObj):
        
    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of ROBUST least squares objective.
        """

        n, d = X.shape

        # Calculate the function value
        f = 0
        for i in range(n):
            # Tip: when you have two terms, it's useful to call them "left" and "right".
            # Believe or not, having two terms show up in your functions is extremely common.
            left = np.exp(w@X[i,:] - y[i])
            right = np.exp(y[i] - w@X[i,:])
            f += np.log(left + right)

        # Calculate the gradient value
        r = np.zeros(n)
        for i in range(n):
            left = np.exp(w@X[i,:] - y[i])
            right = np.exp(y[i] - w@X[i,:])
            r[i] = (left - right) / (left + right)
        g = X.T@r

        return f, g

class FunObjLogReg(FunObj):

    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of logistics regression objective.
        """ 
        Xw = X @ w
        yXw = y * Xw  # element-wise multiply

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T @ res
    
        return f, g

class FunObjLogRegL2(FunObj):

    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of L2-regularized logistics regression objective.
        """ 

        """YOUR CODE HERE FOR Q2.1"""
        # raise NotImplementedError()

        Xw = X @ w
        yXw = y * Xw  # element-wise multiply

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + 0.5 * self.lammy * np.sum(w ** 2)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T @ res + self.lammy * w
    
        return f, g

class FunObjLogRegL0(FunObj):

    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        """
        Evaluates the function value of of L0-regularized logistics regression objective.
        """ 
        Xw = X @ w
        yXw = y * Xw  # element-wise multiply

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + self.lammy * len(w)
        
        # We cannot differentiate the "length" function
        g = None
        return f, g

class FunObjSoftmax(FunObj):

    def evaluate(self, w, X, y):
        n, d = X.shape
        k = len(np.unique(y))

        """YOUR CODE HERE FOR Q3.4"""
        # Hint: you will want to use NumPy's reshape() or flatten()
        # to be consistent with our matrix notation.

        W = w.reshape(k, d)
        G = np.zeros([k, d])
        
        # Precompute dot products
        XW = X @ W.T  # n-by-k matrix, XW[i, c] is the dot product between example i and class c weights
        exp_XW = np.exp(XW) # n-by-k matrix, exp_XW[i, c] is the exponential of dot product between example i and class c weights
        sum_exp_XW = np.sum(exp_XW, axis=1)  # n-by-1 vector, sum_exp_XW[i] is the sum of exponentials of dot products between example i and each class's weights
        log_sum_exp_XW = np.log(sum_exp_XW)  # self-explanatory
        
        # Precompute p
        p = np.zeros([k, n])  # p[c, i] is the softmax probability p(y_i = c | W, x_i).
        for c in range(k):
            for i in range(n):
                p[c, i] = exp_XW[i, c] / sum_exp_XW[i]

        # Compute f value
        f = 0
        for i in range(n):
            left = -XW[i, y[i]]
            right = log_sum_exp_XW[i]
            f += left + right

        # Compute gradient. More vectorized the better
        for c in range(k):
            for j in range(d):
                left = X[:, j]
                right = p[c, :] - (y == c)
                G[c, j] = np.sum(left * right)
        g = G.reshape(-1)
        return f, g