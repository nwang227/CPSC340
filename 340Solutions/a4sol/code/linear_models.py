import numpy as np
from numpy.linalg import solve
from scipy.optimize import approx_fprime
import utils

class LogRegClassifier:
    # Generic Logistic Regression classifier,
    # whose behaviour is selected by the combination of
    # function object and optimizer.
    def __init__(self, fun_obj, optimizer):
        self.fun_obj = fun_obj
        self.optimizer = optimizer
        self.bias_yes = True

        # For debugging and making learning curves
        self.fs = []
        self.nonzeros = []
        self.ws = []

    def optimize(self, w_init, X, y):
        """
        Refactored in A4 for less redundancy. All of the gradient-based classes
        will call self.optimize() to get the optimal parameters.
        
        Perform gradient descent using the optimizer.
        """
        n, d = X.shape

        # Initial guess
        w = np.copy(w_init)

        # Reset the optimizer state and tie it to the new parameters.
        # See optimizers.py for why reset() is useful here.
        self.optimizer.reset()
        self.optimizer.set_parameters(w)
        self.optimizer.set_fun_obj_args(X, y)

        # Collect training information for debugging
        fs = []
        gs = []
        ws = []

        # Use gradient descent to optimize w
        for i in range(1000):
            f, g, w, break_yes = self.optimizer.step()
            fs.append(f)
            gs.append(g)
            ws.append(w)
            if break_yes:
                break

        return w, fs, gs, ws

    def fit(self,X, y):
        """
        Generic fitting subroutine in triplet:
        1. Make initial guess
        2. Check correctness of function object
        3. Use gradient descent to optimize
        """
        n, d = X.shape

        # Initial guess
        w = np.zeros(d)

        # Correctness check
        self.fun_obj.check_correctness(w, X, y)

        # Optimize
        self.w, self.fs, self.gs, self.ws = self.optimize(w, X, y)

    def predict(self, X_hat):
        return np.sign(X_hat@self.w)


class LogRegClassifierForwardSelection(LogRegClassifier):
    """
    A logistic regression classifier that uses forward selection during
    its training subrountine. A gradient-based optimizer as well as an objective function is needed.
    """
    def __init__(self, global_fun_obj, optimizer):
        """
        NOTE: There are two function objects involved:
        1. global_fun_obj: a forward selection criterion to evaluate the feature set
        2. a fun_obj tied to the optimizer
        """
        self.global_fun_obj = global_fun_obj
        self.optimizer = optimizer

    def fit(self, X, y):
        n, d = X.shape
        
        # Maintain the index set. We will start with feature 0 by default.
        selected = set()
        selected.add(0)
        min_loss = np.inf
        old_loss = 0
        best_feature = -1

        # We will hill-climb until a local discrete minimum is found.
        while min_loss != old_loss:
            old_loss = min_loss
            print("Epoch {:d}".format(len(selected)))
            print("Selected feature: {:d}".format(best_feature))
            print("Min Loss: {:.3f}".format(min_loss))

            for j in range(d):
                if j in selected:
                    continue

                selected_new = selected | {j} # tentatively add feature "i" to the selected set

                """YOUR CODE HERE FOR Q2.3"""
                # TODO: Fit the model with 'i' added to the features,
                # then compute the loss and update the min_loss/best_feature
                w = np.zeros(len(selected_new))
                w, fs, gs, ws = self.optimize(w, X[:, list(selected_new)], y)
                loss, _ = self.global_fun_obj.evaluate(w, X[:, list(selected_new)], y)
                if loss < min_loss:
                    min_loss = loss
                    best_feature = j

            selected.add(best_feature)

        self.w = np.zeros(d)
        w_init = np.zeros(len(selected))
        self.w[list(selected)], _, _, _ = self.optimize(w_init, X[:, list(selected)], y)


class LeastSquaresClassifier:
    """
    Uses the normal equations to fit a one-vs-all least squares classifier.
    """

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X_hat):
        return np.argmax(X_hat@self.W.T, axis=1)

class LogRegClassifierOneVsAll(LogRegClassifier):
    """
    Uses a function object and an optimizer.
    """

    def fit(self, X, y):
        """
        NOTE: ensure that y's values are {-1, +1} for logistic regression, not {0, 1}
        """
        n, d = X.shape
        k = len(np.unique(y))  # number of classes in label, assume 0, 1, ..., k-1.

        # Initial guess
        W = np.zeros([k, d])
        
        """YOUR CODE HERE FOR Q3.2"""
        self.fun_obj.check_correctness(np.zeros(d), X, (y == 1).astype(np.float32))
        for c in range(k):
            y_yes = y == c  # reduce classes to binary based on c
            y_no = y != c
            y_binary = np.zeros(n)
            y_binary[y_yes] = 1
            y_binary[y_no] = -1
            w = np.zeros(d)  # initial guess
            w, fs, gs, ws = self.optimize(w, X, y_binary)
            W[c, :] = w

        self.W = W

    def predict(self, X_hat):
        return np.argmax(X_hat@self.W.T, axis=1)

class MulticlassLogRegClassifier(LogRegClassifier):
    """
    LogRegClassifier's extention for multiclass classification.
    The constructor method and optimize() are inherited, so
    all you need to implement are fit() and predict() methods.
    """

    def fit(self, X, y):
        """YOUR CODE HERE FOR Q3.4"""
        # raise NotImplementedError()
        n, d = X.shape
        k = len(np.unique(y))
        w_init = np.zeros(k*d)


        w, fs, gs, ws = self.optimize(w_init, X, y)
        W = w.reshape([k, d])
        self.W = W

    def predict(self, X_hat):
        """YOUR CODE HERE FOR Q3.4"""
        # raise NotImplementedError()
        return np.argmax(X_hat@self.W.T, axis=1)
