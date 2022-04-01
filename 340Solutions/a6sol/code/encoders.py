import numpy as np

"""
Changed name from compressors to encoders in A6.
Contains class definitions related to latent factor models, whose behaviours are
encapsulated by the "learned encoders", which are objects implementing encode() method.
"""

class LinearEncoder:
    """
    Latent factor models that can "encode" X into Z, and "decode" Z into X based on latent factors W.
    """

    def encode(self, X):
        """
        Use the column-wise mean and principal components to
        compute the "component scores" to encode
        """
        X = X - self.mu
        Z = X@self.W.T
        return Z

    def decode(self, Z):
        X_hat = Z@self.W + self.mu
        return X_hat

class LinearEncoderPCA(LinearEncoder):
    """
    Changed name from PCA to PCAEncoder for A6, just to be consistent with rest of classes
    Solves the PCA problem min_Z,W (Z*W-X)^2 using SVD
    """

    def __init__(self, k):
        self.k = k
        self.mu = None
        self.W = None

    def fit(self, X):
        """
        Learns the principal components by delegating to SVD solver.
        "Fitting" here is the matter of populating:
        self.mu: the column-wise mean
        self.W: the principal components
        """
        self.mu = np.mean(X,axis=0)
        X = X - self.mu

        U, s, Vh = np.linalg.svd(X)
        self.W = Vh[:self.k]

class LinearEncoderGradient(LinearEncoder):
    """
    Generic linear encoder that uses a function object and an optimizer
    to select desired properties of its parameters W (a k-by-d matrix).
    Matrix factorization X = Z@W is one way to do linear encoding, but there's
    nothing stopping us from directly optimizing encoded features, e.g. MDS.
    """

    def __init__(self, k, fun_obj_w, fun_obj_z, optimizer_w, optimizer_z):
        """
        Unlike in linear models, we explicitly take function objects for Z and W separately.
        k is the number of factors.
        The function objects are assumed to implement evaluate(parameters, X, k), where k is the number of factors.
        Each function object may use different optimizers as well, e.g. proximal gradient for W to encourage sparsity.
        The optimizers themselves will stay unchanged from previous assignments,
        meaning we must reshape and concatenate our gradients carefully.
        """
        self.k = k
        self.fun_obj_w = fun_obj_w
        self.fun_obj_z = fun_obj_z
        self.optimizer_w = optimizer_w
        self.optimizer_z = optimizer_z
    
    def optimize(self, W_init, Z_init, X, optimize_factors_yes=True, optimize_features_yes=True, switch_every=10):
        """
        Perform optimization to produce:
        1. W, the latent factors
        2. Z, the encoded features

        By default, we use alternating optimization.
        The keyword arguments optimize_factors_yes and optimize_features_yes controls whether
        gradient descent should be invoked for the factors W and/or features Z.
        The keyword argument switch_every controls how many iterations of gradient descent is used
        for optimizing features before moving onto optimizing factors and vice versa.
        """
        n, d = X.shape
        k, _ = W_init.shape

        # Initial guess
        W = np.copy(W_init)
        Z = np.copy(Z_init)

        # Optimizers will work with flattened values by default
        w = W.reshape(-1)
        z = Z.reshape(-1)

        f_w, g_w = self.fun_obj_w.evaluate(w, Z, X)  # Note the arguments here
        f_z, g_z = self.fun_obj_z.evaluate(z, W, X)

        # Collect training information for debugging
        f_ws = [f_w]
        f_zs = [f_z]
        g_ws = [g_w]
        g_zs = [g_z]

        ws = []
        zs = []

        # Use gradient descent to optimize parameters
        break_yes_w = not optimize_factors_yes
        break_yes_z = not optimize_features_yes
        print("Iteration {:d}\tf_w:{:f}\tf_z:{:f}".format(0, f_w, f_z))
        for gd_iteration in range(10):
            # It's hard to determine the true convergence of alternating optimization with the current code,
            # so we will run the outer loop for 10 iterations without worrying about convergence.
            # We still use "local convergence", e.g. whether we reach local minima for W for a fixed Z, etc.

            if optimize_factors_yes:
                # Optimize W using latest Z
                Z = z.reshape(n, k)
                self.optimizer_w.reset()
                self.optimizer_w.set_fun_obj(self.fun_obj_w)
                self.optimizer_w.set_fun_obj_args(Z, X)
                self.optimizer_w.set_parameters(w)
                for _ in range(switch_every):
                    f_w, g_w, w, break_yes_w = self.optimizer_w.step()
                    f_ws.append(f_w)
                    g_ws.append(g_w)
                    ws.append(w)
                    if break_yes_w:
                        break

            if optimize_features_yes:
                # Optimize Z using latest W
                W = w.reshape(k, d)
                self.optimizer_z.reset()
                self.optimizer_z.set_fun_obj(self.fun_obj_z)
                self.optimizer_z.set_fun_obj_args(W, X)
                self.optimizer_z.set_parameters(z)
                for _ in range(switch_every):
                    f_z, g_z, z, break_yes_z = self.optimizer_z.step()
                    f_zs.append(f_z)
                    g_zs.append(g_z)
                    zs.append(z)
                    if break_yes_z:
                        break

            print("Iteration {:d}\tf_w:{:f}\tf_z:{:f}".format(gd_iteration + 1, f_w, f_z))

        # parse the parameters to get optimized W and Z
        W = w.reshape(k, d)
        Z = z.reshape(n, k)

        return W, Z, f_ws, f_zs, g_ws, g_zs, ws, zs

    def fit(self, X):
        """
        Perform optimization to produce: 
        1. Z, the encoded features
        2. W, the latent factors
        """
        n, d = X.shape

        # Center X
        self.mu = np.mean(X, axis=0)
        X_centered = X - self.mu

        # As discussed in class, random initialization is important to avoid unstable optima
        W_init = np.random.randn(self.k, d)
        Z_init = np.random.randn(n, self.k)
        
        W, Z, f_ws, f_zs, g_ws, g_zs, ws, zs = self.optimize(W_init, Z_init, X_centered)
        self.W = W
        self.Z = Z

    def encode(self, X, max_evals=100):
        """
        We do not enforce that W is orthogonal, so we optimize Z
        """
        n, d = X.shape
        X_hat_centered = X - self.mu
        Z = np.zeros([n, self.k])
        z = Z.reshape(-1)
        self.optimizer_z.reset()
        self.optimizer_z.set_fun_obj(self.fun_obj_z)
        self.optimizer_z.set_fun_obj_args(self.W, X_hat_centered)
        self.optimizer_z.set_parameters(z)
        
        for _ in range(max_evals):
            f_z, g_z, z, break_yes_z = self.optimizer_z.step()
            if break_yes_z:
                break

        Z = z.reshape(n, self.k)

        return Z

class FactorlessEncoder:
    """
    As opposed to linear encoders, factorless encoders will use 
    optimization and distance-based methods to encode input examples.
    """

    def encode(self, X):
        raise NotImplementedError()

class FactorlessEncoderGradient(FactorlessEncoder):
    """
    A factorless encoder that uses gradient descent to optimize learned features.
    k is not the number of factors anymore, since we are not keeping them.
    Rather, k is the dimensionality of the learned features.
    """
    def __init__(self, k, fun_obj, optimizer):
        self.k = k
        self.fun_obj = fun_obj
        self.optimizer = optimizer

    def fit(self, X):
        raise RuntimeError("Nothing to fit!")

    def encode(self, X, max_evals=100):
        n, d = X.shape
        Z = self.get_initial_features(X)
        z = Z.reshape(-1)

        self.optimizer.reset()
        self.optimizer.set_fun_obj(self.fun_obj)
        self.optimizer.set_fun_obj_args(X)
        self.optimizer.set_parameters(z)

        self.fs = []

        for _ in range(max_evals):
            f, g, z, break_yes = self.optimizer.step()
            if break_yes:
                break
        
        Z = z.reshape(n, self.k)
        return Z

    def get_initial_features(self, X):
        n, d = X.shape
        return np.random.randn(n, self.k)

class FactorlessEncoderGradientWarmStart(FactorlessEncoderGradient):
    """
    A factorless encoder that uses gradient descent to optimize learned features.
    It also receives a "warm-start encoder" which will provide the initial Z,
    as opposed to having to randomly initialize.
    """

    def __init__(self, k, fun_obj, optimizer, warm_start_encoder):
        super().__init__(k, fun_obj, optimizer)
        self.fun_obj = fun_obj
        self.optimizer = optimizer
        self.warm_start_encoder = warm_start_encoder

    def fit(self, X):
        self.warm_start_encoder.fit(X)

    def get_initial_features(self, X):
        return self.warm_start_encoder.encode(X)

class NonLinearEncoderMatMulActivation:
    """
    An "encoder" object encapsulating matrix multiplication and non-linear activation into its encode() term.
    This encoder in itself does not have fit() implemented, but it can be optimized
    as a part of multi-layer perceptron training via backpropagation.
    """

    def __init__(self, input_dim, output_dim, scale=1e-1, activate_yes=True):
        self.W = scale * np.random.randn(output_dim, input_dim)  # by convention we will right-multiply by W.T
        # self.b = scale * np.random.randn(output_dim)
        self.b = np.zeros(output_dim)
        self.activate_yes = activate_yes

    @property
    def size(self):
        return self.W.size + self.b.size

    def set_weights_and_biases(self, w):
        """
        Take a flattened parameter vector w and take it as weights and biases
        """
        assert w.size == self.W.size + self.b.size
        weights_flat = w[:-self.b.size]
        biases_flat = w[-self.b.size:]
        self.W = weights_flat.reshape(self.W.shape)
        self.b = biases_flat.reshape(self.b.shape)

    def get_parameters_flattened(self):
        return np.concatenate([self.W.reshape(-1), self.b.reshape(-1)])

    def encode(self, X):
        """
        Assume that X is an n-by-input_dim array, then we output a Z, an n-by-output_dim array.
        """
        Z = (X @ self.W.T) + self.b
        if self.activate_yes:
            return 1. / (1 + np.exp(-Z))
            # return np.maximum(Z, 0)
        else:
            return Z

    def fit(self, X):
        raise RuntimeError("fit() should not be called.")

class NonLinearEncoderMultiLayer:
    """
    Encoder architecture used by fully-connected neural networks.
    Under the hood, we instantiate multiple NonLinearEncoderMatMulActivation instances and chain them together.
    layer_sizes is a list of integers specifying the input and output dimensions.
    Important: the first number should be 'd', the number of features.
    
    fit() is not implemented as the encoder parameters should be
    populated by a neural network's optimization via backpropagation.
    """

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights_and_biases = None

        # Instantiate "child encoders" that will be chained together
        self.encoders = []
        for input_dim, output_dim in zip(layer_sizes, layer_sizes[1:]):  # zip() trick to iterate consecutive pairs
            encoder = NonLinearEncoderMatMulActivation(input_dim, output_dim)
            self.encoders.append(encoder)

    @property
    def size(self):
        return self.get_parameters_flattened().size

    def set_weights_and_biases(self, w):
        """
        Receive parameters in a vector form. Parse into weights and biases.
        """
        encoder_sizes = [encoder.size for encoder in self.encoders]
        assert w.size == np.sum(encoder_sizes)
        ws = np.array_split(w, np.cumsum(encoder_sizes))  # partition w into sub-arrays, corresponding to each encoder's size
        for w_for_encoder, encoder in zip(ws, self.encoders):
            encoder.set_weights_and_biases(w_for_encoder)

    def get_parameters_flattened(self):
        return np.concatenate([encoder.get_parameters_flattened() for encoder in self.encoders])

    def encode(self, X):
        """
        Multi-layer encoder's activations are used for gradient computation, so we will collect and return them.
        """
        activations = [X]
        for encoder in self.encoders:
            Z = encoder.encode(X)
            activations.append(Z)
            X = Z
        return Z, activations[:-1]

    def fit(self, X):
        raise RuntimeError("fit() should not be called.")





