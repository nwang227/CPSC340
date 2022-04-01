import numpy as np


# helper functions to transform between one big vector of weights
# and a list of layer parameters of the form (W,b)
def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights, ())])


def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes) - 1):
        W_size = layer_sizes[i + 1] * layer_sizes[i]
        b_size = layer_sizes[i + 1]

        W = np.reshape(
            weights_flat[counter : counter + W_size],
            (layer_sizes[i + 1], layer_sizes[i]),
        )
        counter += W_size

        b = weights_flat[counter : counter + b_size][None]
        counter += b_size

        weights.append((W, b))
    return weights


class NeuralNet:
    """
    A neural network is an encoder and a linear model trained at the same time.
    encode() uses the "multi-layer perceptron" method,
    i.e. matrix multiplication followed by non-linear element-wise operation
    to map X, input examples from raw feature space, to Z, examples in new features Z.
    """

    # uses sigmoid nonlinearity (it's hard to make it modular without autodiff!)
    def __init__(self, loss_fn, optimizer, encoder, predictor, classifier_yes=False):
        """
        loss_fn will be a composition of two function objects:
        one for backpropagation of derivatives into weights and biases and
        one for the final layer prediction.

        For A6, encoder should be a MultiLayerEncoder instance. See encoders.py.
        More sophisticated encoders like convolutional neural networks and recurrent neural networks
        are very tricky to implement without automtaic differentiation, which is our topic for A7.
        """
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.encoder = encoder
        self.predictor = predictor
        self.classifier_yes = classifier_yes  # if true, prediction will be discrete

    def fit(self, X, y):
        # Generally handle multi-output y, which can be an n-by-k matrix instead of an n-by-1 vector
        if y.ndim == 1:
            y = y[:, None]

        # Initial guess
        w_init = np.concatenate(
            [
                self.encoder.get_parameters_flattened(),
                self.predictor.get_parameters_flattened(),
            ]
        )
        f, g = self.loss_fn.evaluate(w_init, X, y)

        # utils.check_gradient(self, X, y, len(weights_flat), epsilon=1e-6)
        # weights_flat_new, f = findMin.findMin(self.funObj, weights_flat, self.max_iter, X, y, verbose=True)

        self.optimizer.reset()
        self.optimizer.set_fun_obj(self.loss_fn)
        self.optimizer.set_fun_obj_args(X, y)
        self.optimizer.set_parameters(w_init)

        self.fs = [f]
        self.gs = [g]
        for i in range(1000):
            f, g, w, break_yes = self.optimizer.step()
            if (
                break_yes
            ):  # in reality, convergence will rarely happen, so we will ignore break conditions
                break

        predictor_size = self.predictor.size
        w_encoder = w[:-predictor_size]
        w_predictor = w[-predictor_size:]
        self.encoder.set_weights_and_biases(w_encoder)
        self.predictor.set_weights_and_biases(w_predictor)

    def get_parameters_flattened(self):
        return np.concatenate(
            [
                self.encoder.get_parameters_flattened(),
                self.predictor.get_parameters_flattened(),
            ]
        )

    def predict(self, X):
        """
        Prediction in neural networks is in two steps:
        1. Encode
        2. Predict
        In case we deal with multi-class classification situation, we take argmax of the linear output.
        """
        Z, _ = self.encoder.encode(
            X
        )  # presumably MultiLayerEncoder, which also returns activation trajectory.
        y_hat = self.predictor.predict(Z)
        if self.classifier_yes:
            return np.argmax(y_hat, axis=1)  # multi-class situation
        else:
            return y_hat

    def encode(self, X):
        return self.encoder.encode(X)
