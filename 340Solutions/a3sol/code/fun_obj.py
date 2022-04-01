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
        """
        raise NotImplementedError

    def check_correctness(self, w, X, y):
        n, d = X.shape
        estimated_gradient = approx_fprime(w.flatten(), lambda w: self.evaluate(w.reshape((d,1)),X,y)[0], epsilon=1e-6)
        implemented_gradient = self.evaluate(w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
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

        """YOUR CODE HERE FOR Q2.3"""
        # raise NotImplementedError()
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