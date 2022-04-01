from linear_models import LinearModelMultiOutput, MulticlassLogRegClassifier
from learning_rate_getters import LearningRateGetterConstant, LearningRateGetterInverseSqrt
from fun_obj import FunObjMDSEuclidean, FunObjMDSGeodesic, FunObjMLPLogRegL2, FunObjPCAFactors, FunObjPCAFeatures, FunObjRobustPCAFactors, FunObjRobustPCAFeatures, FunObjSoftmax
import os
import pickle
import gzip
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer

from neural_net import NeuralNet
from optimizers import OptimizerGradientDescent, OptimizerGradientDescentHeavyBall, OptimizerGradientDescentLineSearch, OptimizerStochasticGradient

from encoders import FactorlessEncoderGradient, FactorlessEncoderGradientWarmStart, LinearEncoderGradient, LinearEncoderPCA, NonLinearEncoderMultiLayer
import utils

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        X_train = load_dataset('highway.pkl')['X'].astype(float) / 255.
        n, d = X_train.shape
        h, w = 64, 64  # height and width of each image
        k = 5  # number of PCs
        threshold = 0.1  # threshold for being considered "foreground"

        # PCA with SVD
        model = LinearEncoderPCA(k)
        model.fit(X_train)
        Z = model.encode(X_train)
        X_hat = model.decode(Z)

        # PCA with alternating minimization
        fun_obj_w = FunObjPCAFactors()
        fun_obj_z = FunObjPCAFeatures()
        optimizer_w = OptimizerGradientDescentLineSearch(max_evals=100, verbose=False)
        optimizer_z = OptimizerGradientDescentLineSearch(max_evals=100, verbose=False)
        model = LinearEncoderGradient(k, fun_obj_w, fun_obj_z, optimizer_w, optimizer_z)
        model.fit(X_train)
        Z_alt = model.encode(X_train)
        X_hat_alt = model.decode(Z_alt)

        fig, ax = plt.subplots(2,3)
        for i in range(10):
            ax[0,0].set_title('$X$')
            ax[0,0].imshow(X_train[i].reshape(h,w).T, cmap='gray')

            ax[0,1].set_title('$\hat{X}$ (L2)')
            ax[0,1].imshow(X_hat[i].reshape(h,w).T, cmap='gray')
            
            ax[0,2].set_title('$|x_i-\hat{x_i}|$>threshold (L2)')
            ax[0,2].imshow((np.abs(X_train[i] - X_hat[i])<threshold).reshape(h,w).T, cmap='gray')

            ax[1,0].set_title('$X$')
            ax[1,0].imshow(X_train[i].reshape(h,w).T, cmap='gray')
            
            ax[1,1].set_title('$\hat{X}$ (L1)')
            ax[1,1].imshow(X_hat_alt[i].reshape(h,w).T, cmap='gray')

            ax[1,2].set_title('$|x_i-\hat{x_i}|$>threshold (L1)')
            ax[1,2].imshow((np.abs(X_train[i] - X_hat_alt[i])<threshold).reshape(h,w).T, cmap='gray')

            utils.savefig('highway_{:03d}.jpg'.format(i))

    elif question == "1.1":
        X_train = load_dataset('highway.pkl')['X'].astype(float) / 255.
        n, d = X_train.shape
        h, w = 64, 64  # height and width of each image
        k = 5  # number of PCs
        threshold = 0.1  # threshold for being considered "foreground"

        # PCA with SVD
        model = LinearEncoderPCA(k)
        model.fit(X_train)
        Z = model.encode(X_train)
        X_hat = model.decode(Z)

        # TODO: Implement function objects for robust PCA in fun_obj.py
        fun_obj_w = FunObjRobustPCAFactors(1e-6)
        fun_obj_z = FunObjRobustPCAFeatures(1e-6)
        optimizer_w = OptimizerGradientDescentLineSearch(max_evals=100, verbose=False)
        optimizer_z = OptimizerGradientDescentLineSearch(max_evals=100, verbose=False)
        model = LinearEncoderGradient(k, fun_obj_w, fun_obj_z, optimizer_w, optimizer_z)
        model.fit(X_train)
        Z_alt = model.encode(X_train)
        X_hat_alt = model.decode(Z_alt)

        fig, ax = plt.subplots(2,3)
        for i in range(10):
            ax[0,0].set_title('$X$')
            ax[0,0].imshow(X_train[i].reshape(h,w).T, cmap='gray')

            ax[0,1].set_title('$\hat{X}$ (L2)')
            ax[0,1].imshow(X_hat[i].reshape(h,w).T, cmap='gray')
            
            ax[0,2].set_title('$|x_i-\hat{x_i}|$>threshold (L2)')
            ax[0,2].imshow((np.abs(X_train[i] - X_hat[i])<threshold).reshape(h,w).T, cmap='gray')

            ax[1,0].set_title('$X$')
            ax[1,0].imshow(X_train[i].reshape(h,w).T, cmap='gray')
            
            ax[1,1].set_title('$\hat{X}$ (L1)')
            ax[1,1].imshow(X_hat_alt[i].reshape(h,w).T, cmap='gray')

            ax[1,2].set_title('$|x_i-\hat{x_i}|$>threshold (L1)')
            ax[1,2].imshow((np.abs(X_train[i] - X_hat_alt[i])<threshold).reshape(h,w).T, cmap='gray')

            utils.savefig('highway_{:03d}.jpg'.format(i))

    elif question == "2":
        dataset = load_dataset('animals.pkl')
        X_train = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X_train.shape
        print("n =", n)
        print("d =", d)

        f1, f2 = np.random.choice(d, size=2, replace=False)

        plt.figure()
        plt.scatter(X_train[:,f1], X_train[:,f2])
        plt.xlabel("$x_{:d}$".format(f1))
        plt.ylabel("$x_{:d}$".format(f2))
        for i in range(n):
            plt.annotate(animals[i], (X_train[i,f1], X_train[i,f2]))
        
        utils.savefig('animals_two_random_features.png')

    elif question == "2.1":
        dataset = load_dataset('animals.pkl')
        X_train = dataset['X'].astype(float)
        animals = dataset['animals']
        n, d = X_train.shape

        # No warm-start version
        fun_obj = FunObjMDSEuclidean(X_train)  # Populates the Euclidean distance matrix based on X, for fast re-use.
        optimizer = OptimizerGradientDescentLineSearch()
        model = FactorlessEncoderGradient(2, fun_obj, optimizer)
        Z = model.encode(X_train)
        f, g = fun_obj.evaluate(Z.reshape(-1), X_train)
        print("MDS without warm-start: {:f}".format(f))

        plt.figure()
        plt.scatter(Z[:,0], Z[:,1])
        plt.xlabel("$z_{i1}$")
        plt.ylabel("$z_{i2}$")
        for i in range(n):
            plt.annotate(animals[i], (Z[i,0], Z[i,1]))
        
        utils.savefig('animals_mds_features.png')

        """YOUR CODE HERE FOR Q2.1"""
        # TODO: use PCA as a warm-start encoder
        # model = FactorlessEncoderGradientWarmStart(2, fun_obj, optimizer, warm_start_encoder)

        fun_obj = FunObjMDSEuclidean(X_train)  # Populates the Euclidean distance matrix based on X, for fast re-use.
        optimizer = OptimizerGradientDescentLineSearch()
        warm_start_encoder = LinearEncoderPCA(2)
        model = FactorlessEncoderGradientWarmStart(2, fun_obj, optimizer, warm_start_encoder)
        model.fit(X_train)
        Z = model.encode(X_train)
        f, g = fun_obj.evaluate(Z.reshape(-1), X_train)
        print("MDS with warm-start: {:f}".format(f))

        model = LinearEncoderPCA(2)
        model.fit(X_train)
        Z = model.encode(X_train)
        f, g = fun_obj.evaluate(Z.reshape(-1), X_train)
        print("PCA: {:f}".format(f))
        
    elif question == "2.2":
        dataset = load_dataset('animals.pkl')
        X_train = dataset['X'].astype(float)
        animals = dataset['animals']
        n, d = X_train.shape

        """YOUR CODE HERE FOR Q2.2"""
        for n_neighbours in [2, 3]:
            fun_obj = FunObjMDSGeodesic(X_train, n_neighbours)  # Populates the geodesic distance matrix based on X, for fast re-use.
            optimizer = OptimizerGradientDescentLineSearch()
            warm_start_encoder = LinearEncoderPCA(2)
            model = FactorlessEncoderGradientWarmStart(2, fun_obj, optimizer, warm_start_encoder)
            model.fit(X_train)
            Z = model.encode(X_train)

            print(fun_obj.evaluate(Z.reshape(-1), X_train)[0])

            plt.figure()
            plt.scatter(Z[:,0], Z[:,1])
            plt.xlabel("$z_{i1}$")
            plt.ylabel("$z_{i2}$")
            plt.title('ISOMAP with NN={:d}'.format(n_neighbours))
            for i in range(n):
                plt.annotate(animals[i], (Z[i,0], Z[i,1]))
            
            utils.savefig('animals_isomap_features_{:d}nn.png'.format(n_neighbours))

    elif question == '2.3':
        dataset = load_dataset('animals.pkl')
        X_train = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X_train.shape

        """YOUR CODE HERE FOR Q2.3"""
        # standardize columns
        X_train = utils.standardize_cols(X_train)

        model = TSNE()
        Z = model.fit_transform(X_train)
        fig, ax = plt.subplots()
        ax.scatter(Z[:,0], Z[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('T-SNE')
        for i in range(n):
            ax.annotate(animals[i], (Z[i,0], Z[i,1]))

        utils.savefig('TSNE_animals.png')

    elif question == "3":

        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

        # Use these for softmax classifier
        X_train, y_train = train_set
        X_valid, y_valid = valid_set

        binarizer = LabelBinarizer()
        Y_train = binarizer.fit_transform(y_train)

        n, d = X_train.shape
        _, k = Y_train.shape  # k is the number of classes

        fun_obj = FunObjSoftmax(k)
        child_optimizer = OptimizerGradientDescent()
        learning_rate_getter = LearningRateGetterConstant(1e-3)
        optimizer = OptimizerStochasticGradient(child_optimizer, learning_rate_getter, 500, max_evals=10)
        model = MulticlassLogRegClassifier(fun_obj, optimizer)

        t = time.time()
        model.fit(X_train, y_train)
        print("Fitting took {:f} seconds".format((time.time()-t)))
        
        # Comput training error
        y_hat = model.predict(X_train)
        err_train = np.mean(y_hat != y_train)
        print("Training error = ", err_train)
        
        # Compute validation error
        y_hat = model.predict(X_valid)
        err_valid = np.mean(y_hat != y_valid)
        print("Validation error     = ", err_valid)

    elif question == "3.2":

        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

        # Use these for softmax classifier
        X_train, y_train = train_set
        X_valid, y_valid = valid_set

        # Use these for training our MLP classifier
        binarizer = LabelBinarizer()
        Y_train = binarizer.fit_transform(y_train)

        n, d = X_train.shape
        _, k = Y_train.shape  # k is the number of classes

        X_train_standardized, mu, sigma = utils.standardize_cols(X_train)
        X_valid_standardized, _, _ = utils.standardize_cols(X_valid, mu, sigma)

        # Assemble a neural network
        hidden_feature_dims = [256]  # put hidden layer dimensions to increase the number of layers in encoder
        output_dim = 256

        # First, initialize an encoder and a predictor
        layer_sizes = [d, *hidden_feature_dims, output_dim]
        encoder = NonLinearEncoderMultiLayer(layer_sizes)
        predictor = LinearModelMultiOutput(output_dim, k)

        # Function object will associate the encoder and the predictor during training
        fun_obj = FunObjMLPLogRegL2(encoder, predictor, 0)

        # Choose optimization strategy
        child_optimizer = OptimizerGradientDescentHeavyBall(0.0)
        # child_optimizer = OptimizerGradientDescent()
        # learning_rate_getter = LearningRateGetterConstant(1e-2)
        learning_rate_getter = LearningRateGetterInverseSqrt(1e-3)
        optimizer = OptimizerStochasticGradient(child_optimizer, learning_rate_getter, 500, max_evals=20)

        # Assemble!
        model = NeuralNet(fun_obj, optimizer, encoder, predictor, classifier_yes=True)

        t = time.time()
        model.fit(X_train_standardized, Y_train)
        print("Fitting took {:f} seconds".format((time.time()-t)))

        # Comput training error
        y_hat = model.predict(X_train_standardized)
        err_train = np.mean(y_hat != y_train)
        print("Training error = ", err_train)
        
        # Compute validation error
        y_hat = model.predict(X_valid_standardized)
        err_valid = np.mean(y_hat != y_valid)
        print("Validation error     = ", err_valid)


    elif question == "3.3":
        
        data = load_dataset("sinusoids.pkl")
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_valid = data["X_valid"]
        y_valid = data["y_valid"]

        n, d = X_train.shape
        k = len(np.unique(y_train))

        Y_train = np.stack([1 - y_train, y_train], axis=1).astype(np.uint)

        plt.figure()
        plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], color="b", label="class 0")
        plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], color="r", label="class 1")
        plt.title("Sinusoid data, non-convex but separable.")
        utils.savefig("sinusoids.png")
        
        X_train_standardized, mu, sigma = utils.standardize_cols(X_train)
        X_valid_standardized, _, _ = utils.standardize_cols(X_valid, mu, sigma)

        hidden_feature_dims = [2, 2]  # put hidden layer dimensions to increase the number of layers in encoder
        output_dim = 2
        layer_sizes = [d, *hidden_feature_dims, output_dim]

        best_err_valid = np.inf
        best_model = None
        for seed in range(100):  # "grid search over random seeds"
            # First, initialize an encoder and a predictor
            encoder = NonLinearEncoderMultiLayer(layer_sizes)
            predictor = LinearModelMultiOutput(output_dim, k)
            fun_obj = FunObjMLPLogRegL2(encoder, predictor, 0.)
            optimizer = OptimizerGradientDescentLineSearch()
            model = NeuralNet(fun_obj, optimizer, encoder, predictor, classifier_yes=True)
            for _ in range(10):  # "continual warm-start with resets", one brute-force method to fight NP-hard problems!  
                # calling fit() will reset the optimizer state, but the encoder and predictor's parameters will stay intact.
                model.fit(X_train_standardized, Y_train)
            
                # Comput training error
                y_hat = model.predict(X_train_standardized)
                err_train = np.mean(y_hat != y_train)
                
                # Compute validation error
                y_hat = model.predict(X_valid_standardized)
                err_valid = np.mean(y_hat != y_valid)

                if err_valid < best_err_valid:
                    best_err_valid = err_valid
                    best_model = model

                    print("Training error = ", err_train)
                    print("Validation error     = ", err_valid)

        # Visualize learned features
        Z_train, _ = best_model.encode(X_train_standardized)

        plt.figure()
        plt.scatter(Z_train[y_train==0, 0], Z_train[y_train==0, 1], color="b", label="class 0")
        plt.scatter(Z_train[y_train==1, 0], Z_train[y_train==1, 1], color="r", label="class 1")
        plt.xlabel("$z_{i1}$")
        plt.ylabel("$z_{i2}$")
        plt.title("Learned features of sinusoid data\n hidden dimensions={:s}, output dimension={:d}".format(str(hidden_feature_dims), output_dim))
        utils.savefig("sinusoids_learned_features_{:s}_{:d}.png".format(str(hidden_feature_dims), output_dim))

        utils.plot_classifier(best_model, X_train_standardized, y_train)
        plt.title("Decision boundary in original feature space\nhidden dimensions={:s}, output dimension={:d}".format(str(hidden_feature_dims), output_dim))
        utils.savefig("sinusoids_decision_boundary_{:s}_{:d}.png".format(str(hidden_feature_dims), output_dim))

    else:
        print("Unknown question: %s" % question)    