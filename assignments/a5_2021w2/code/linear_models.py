import numpy as np

"""
Contains class definitions for linear supervised models.
"""


class LinearModel:
    """
    Generic linear model, supporting generic loss functions (FunObj subclasses)
    and optimizers.

    See optimizers.py for optimizers.
    See fun_obj.py for loss function objects, which must implement evaluate()
    and return f and g values corresponding to current parameters.
    """

    def __init__(self, loss_fn, optimizer, check_correctness=False):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.bias_yes = True
        self.check_correctness = check_correctness

        # For debugging and making learning curves
        self.fs = []
        self.nonzeros = []
        self.ws = []

    def optimize(self, w_init, X, y):
        "Perform gradient descent using the optimizer."
        n, d = X.shape

        # Initial guess
        w = np.copy(w_init)
        f, g = self.loss_fn.evaluate(w, X, y)

        # Reset the optimizer state and tie it to the new parameters.
        # See optimizers.py for why reset() is useful here.
        self.optimizer.reset()
        self.optimizer.set_fun_obj(self.loss_fn)
        self.optimizer.set_parameters(w)
        self.optimizer.set_fun_obj_args(X, y)

        # Collect training information for debugging
        fs = [f]
        gs = [g]
        ws = []

        # Use gradient descent to optimize w
        while True:
            f, g, w, break_yes = self.optimizer.step()
            fs.append(f)
            gs.append(g)
            ws.append(w)
            if break_yes:
                break

        return w, fs, gs, ws

    def fit(self, X, y):
        """
        Generic fitting subroutine:
        1. Make initial guess
        2. Check correctness of function object
        3. Use gradient descent to optimize
        """
        n, d = X.shape

        # Correctness check
        if self.check_correctness:
            w = np.random.rand(d)
            self.loss_fn.check_correctness(w, X, y)

        # Initial guess
        w = np.zeros(d)

        # Optimize
        self.w, self.fs, self.gs, self.ws = self.optimize(w, X, y)

    def predict(self, X):
        """
        By default, implement linear regression prediction
        """
        return X @ self.w


class LinearClassifier(LinearModel):
    def predict(self, X_pred):
        return np.sign(X_pred @ self.w)


class KernelClassifier(LinearClassifier):
    def __init__(self, loss_fn, optimizer, kernel, check_correctness=False):
        """
        Make sure loss_fn is kernel-compatible!
        """
        super().__init__(loss_fn, optimizer, check_correctness=check_correctness)
        self.kernel = kernel

    def fit(self, X, y):
        """
        For any choice of kernel, evaluate the Gram matrix first,
        and then run logistic regression.
        """
        self.X = X
        K = self.kernel(X, self.X)
        super().fit(K, y)

    def predict(self, X_pred):
        if self.X is None:
            raise ValueError("Must run fit() before predict()")

        K_pred = self.kernel(X_pred, self.X)
        return super().predict(K_pred)
