from numpy.linalg.linalg import norm
from utils import euclidean_dist_squared, shortest_dist
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

def log_1_plus_exp_safe(x):
    out = np.log(1+np.exp(x))
    out[x > 100] = x[x>100]
    out[x < -100] = np.exp(x[x < -100])
    return out

# Neural network helpers
def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights,())])

def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes)-1):
        W_size = layer_sizes[i+1] * layer_sizes[i]
        b_size = layer_sizes[i+1]
        
        W = np.reshape(weights_flat[counter:counter+W_size], (layer_sizes[i+1], layer_sizes[i]))
        counter += W_size

        b = weights_flat[counter:counter+b_size][None]
        counter += b_size

        weights.append((W,b))  
    return weights

def log_sum_exp(Z):
    Z_max = np.max(Z, axis=1)
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:,None]), axis=1)) # per-colmumn max

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

    def check_correctness(self, w, X, y, epsilon=1e-6):
        n, d = X.shape
        estimated_gradient = approx_fprime(w, lambda w: self.evaluate(w, X, y)[0], epsilon=epsilon)
        _, implemented_gradient = self.evaluate(w, X, y)
        difference = estimated_gradient - implemented_gradient

        # Check the gradient
        if np.max(np.abs(difference))/np.linalg.norm(estimated_gradient) > 1e-6:
            raise Exception('User and numerical derivatives differ:\n%s\n%s' %
                (estimated_gradient[:], implemented_gradient[:]))
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

class FunObjLeastSquaresL2(FunObj):
    
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of least squares objective.
        Least squares objective is the sum of squared residuals.
        """
        n, d = X.shape

        # Prediction is linear combination
        y_hat = X@w
        # Residual is difference between prediction and ground truth
        residuals = y_hat - y
        # Squared residuals gives us the objective function value
        f = 0.5 * np.sum(residuals ** 2) + 0.5 * self.lammy * np.sum(w ** 2)
        # Analytical gradient, written in mathematical form first
        # and then translated into Python
        g = (X.T@X + self.lammy * np.eye(d)) @ w - X.T@y
        return f, g

class FunObjLeastSquaresMultiOutput(FunObj):
    
    def evaluate(self, w, X, Y):
        """
        Generalizes least squares error using Frobenius norm.
        Y is now an n-by-k matrix. Hence, W is a k-by-d matrix.
        """
        n, d = X.shape
        _, k = Y.shape
        W = w.reshape(k, d)
        y_hat = X@W.T
        residuals = y_hat - Y
        f = 0.5 * np.sum(residuals ** 2)
        g = X.T@X@W.T - X.T@Y
        return f, g.T.flatten()

class FunObjLeastSquaresMultiOutputL2(FunObj):
    
    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, Y):
        """
        Generalizes least squares error using Frobenius norm.
        Y is now an n-by-k matrix. Hence, W is a k-by-d matrix.
        """
        n, d = X.shape
        _, k = Y.shape
        W = w.reshape(k, d)
        y_hat = X@W.T
        residuals = y_hat - Y
        f = 0.5 * np.sum(residuals ** 2) + 0.5 * self.lammy * np.sum(W ** 2)
        g = X.T@X@W.T - X.T@Y + self.lammy * W.T
        return f, g.T.flatten()

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
        yXw = np.clip(yXw, -100, 100)  # safeguarding

        # Calculate the function value
        # f = np.sum(np.log(1. + np.exp(-yXw)))
        f = np.sum(log_1_plus_exp_safe(-yXw))

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
        Xw = X @ w
        yXw = y * Xw  # element-wise multiply
        
        # Calculate the function value
        f = np.sum(log_1_plus_exp_safe(-yXw)) + 0.5 * self.lammy * np.sum(w ** 2)
        
        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T @ res + self.lammy * w
    
        return f, g

class FunObjLogRegL2Kernel(FunObj):

    def __init__(self, lammy):
        self.lammy = lammy

    def evaluate(self, w, X, y):
        """
        Evaluates the function and gradient of L2-regularized logistics regression objective.
        """ 
        Xw = X @ w
        yXw = y * Xw  # element-wise multiply
        
        # Calculate the function value
        f = np.sum(log_1_plus_exp_safe(-yXw)) + 0.5 * self.lammy * w.T @ X @ w
        
        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T @ res + self.lammy * X @ w
    
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

    def __init__(self, n_classes):
        self.n_classes = n_classes  # need to be communicated for stochastic gradient etc.

    def evaluate(self, w, X, y):
        """YOUR CODE HERE FOR Q3.4"""
        # Hint: you will want to use NumPy's reshape() or flatten()
        # to be consistent with our matrix notation.
        n, d = X.shape
        k = self.n_classes

        W = w.reshape(k,d)

        y_binary = np.zeros((n, k)).astype(bool)
        y_binary[np.arange(n), y] = 1

        XW = np.dot(X, W.T)
        Z = np.sum(np.exp(XW), axis=1)

        # Calculate the function value
        f = - np.sum(XW[y_binary] - np.log(Z))

        # Calculate the gradient value
        g = (np.exp(XW) / Z[:,None] - y_binary).T@X

        return f, g.flatten()

class FunObjPCAFeatures(FunObj):
    """
    Evaluates PCA objective function and its gradient with respect to Z, the learned features
    """

    def evaluate(self, z, W, X):
        n, d = X.shape
        k, _ = W.shape
        Z = z.reshape(n, k)

        R = Z@W - X
        f = np.sum(R**2)/2
        g = R@W.T
        return f, g.flatten()

class FunObjPCAFactors(FunObj):
    """
    Evaluates PCA objective function and its gradient with respect to W, the learned features
    """

    def evaluate(self, w, Z, X):
        n, d = X.shape
        _, k = Z.shape
        W = w.reshape(k, d)
        
        R = Z@W - X
        f = np.sum(R**2)/2
        g = Z.T@R
        return f, g.flatten()

class FunObjRobustPCAFeatures(FunObj):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def evaluate(self, z, W, X):
        n, d = X.shape
        k, _ = W.shape
        Z = z.reshape(n, k)
        
        R = Z @ W - X
        V = np.sqrt(R**2 + self.epsilon)
        f = np.sum(V)
        dR = R / V
        g = dR @ W.T
        return f, g.flatten()

class FunObjRobustPCAFactors(FunObj):
    
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def evaluate(self, w, Z, X):
        n, d = X.shape
        _, k = Z.shape
        W = w.reshape(k, d)

        R = Z @ W - X
        V = np.sqrt(R**2 + self.epsilon)
        f = np.sum(V)
        dR = R / V
        g = Z.T @ dR
        return f, g.flatten()

class FunObjMDS(FunObj):

    def evaluate(self, z, X):
        """
        Note there is no "W" here, because MDS is a factorless encoder.
        """
        n, d = X.shape
        Z = z.reshape(n, -1)  # k can be deduced by the number of entries in z
        _, k = Z.shape

        # Distance matrix for X
        D = self.D

        f = 0.0
        g = np.zeros((n,k))
        for i in range(n):
            for j in range(i+1,n):
                # Objective Function
                Dz = norm(Z[i]-Z[j])
                s = D[i,j] - Dz
                f = f + (0.5)*(s**2)

                # Gradient
                df = s
                dgi = (Z[i]-Z[j])/Dz
                dgj = (Z[j]-Z[i])/Dz
                g[i] = g[i] - df*dgi
                g[j] = g[j] - df*dgj

        return f, g.flatten()

class FunObjMDSEuclidean(FunObjMDS):

    def __init__(self, X):
        # Pre-compute the (unsquared) Euclidean distance matrix.
        self.D = np.sqrt(euclidean_dist_squared(X, X))

class FunObjMDSGeodesic(FunObjMDS):

    def __init__(self, X, n_neighbours):
        n, d = X.shape
        self.n_neighbours = n_neighbours

        # Pre-compute the geodesic distance matrix, based on Dijkstra's algorithm.
        # Compute Euclidean distances
        D = euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        # Construct nearest neighbour graph
        G = np.zeros([n, n])
        for i in range(n):
            neighbours = np.argsort(D[i])[:self.n_neighbours + 1]
            for j in neighbours:
                G[i,j] = D[i,j]
                G[j,i] = D[j,i]

        # Compute ISOMAP distances
        D = shortest_dist(G)
        
        # If two points are disconnected (distance is Inf)
        # then set their distance to the maximum
        # distance in the graph, to encourage them to be far apart.
        D[np.isinf(D)] = D[~np.isinf(D)].max()

        self.D = D

class FunObjMLP(FunObj):
    """
    Function object for generic multi-layer perceptron
    (aka fully-connected artificial neural networks)
    Without automatic differentiation, function objects tend to get hairy because
    there's no straightfoward separation of hierarchy in terms of model behaviour
    and function/gradient calculation.
    """

    def __init__(self, encoder, predictor):
        self.encoder = encoder
        self.predictor = predictor

    def add_regularization(self, f, g):
        return f, g

    def get_final_layer_f_and_g(self, Z, y):
        raise NotImplementedError()

    def evaluate(self, w, X, y):
        n, d = X.shape
        _, k = y.shape

        # Parse weights and biases for the encoder and the predictor
        encoder_size = self.encoder.size
        w_encoder = w[:encoder_size]
        w_predictor = w[encoder_size:]

        self.encoder.set_weights_and_biases(w_encoder)
        self.predictor.set_weights_and_biases(w_predictor)

        # Use the encoder weights to produce Z
        Z, activations = self.encoder.encode(X)

        f, grad = self.get_final_layer_f_and_g(Z, y)

        # Backpropagate by recursion
        # Predictor phase
        grad_W = grad.T @ Z  # w.r.t predictor weights
        grad_b = np.sum(grad, axis=0)  # w.r.t. predictor biases

        g = [(grad_W, grad_b)]
        
        # Last encoder activation
        grad = grad @ self.predictor.W
        grad = grad * (Z * (1 - Z)) # gradient of sigmoid
        # grad = grad * (Z >= 0) # gradient of ReLU

        # Last encoder gradients
        grad_W = grad.T @ activations[-1]
        grad_b = np.sum(grad, axis=0)
        
        g = [(grad_W, grad_b)] + g # insert to start of list

        # Penultimate encoder to first
        for i in range(len(self.encoder.encoders) - 1, 0, -1):  # goes till i=1.
            encoder = self.encoder.encoders[i]
            grad = grad @ encoder.W
            grad = grad * (activations[i] * (1 - activations[i])) # gradient of sigmoid
            # grad = grad * (activations[i] >= 0) # gradient of ReLU
            grad_W = grad.T @ activations[i - 1]
            grad_b = np.sum(grad, axis=0)

            g = [(grad_W, grad_b)] + g # insert to start of list

        g = flatten_weights(g)
        
        # add L2 regularization
        f += 0.5 * self.lammy * np.sum(w**2)
        g += self.lammy * w 
        
        return f, g

class FunObjMLPLeastSquaresL2(FunObjMLP):

    def __init__(self, encoder, predictor, lammy=1.):
        super().__init__(encoder, predictor)
        self.lammy = lammy

    def get_final_layer_f_and_g(self, Z, y):
        # Use the predictor weights with bias to get prediction
        y_hat = self.predictor.predict(Z)

        f = 0.5 * np.sum((y_hat - y)**2)  
        grad = y_hat - y # gradient for L2 loss
        return f, grad

class FunObjMLPLogRegL2(FunObjMLP):
    
    def __init__(self, encoder, predictor, lammy=1.):
        super().__init__(encoder, predictor)
        self.lammy = lammy

    def get_final_layer_f_and_g(self, Z, y):
        y_hat = self.predictor.predict(Z)
        tmp = np.sum(np.exp(y_hat), axis=1)
        # f = -np.sum(yhat[y.astype(bool)] - np.log(tmp))
        f = -np.sum(y_hat[y.astype(bool)] - log_sum_exp(y_hat))
        grad = np.exp(y_hat) / tmp[:, None] - y
        return f, grad
    

