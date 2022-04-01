import argparse
from fun_obj import FunObjLogReg, FunObjLogRegL0, FunObjLogRegL2, FunObjSoftmax
from optimizers import OptimizerGradientDescent, OptimizerGradientDescentLineSearch, OptimizerGradientDescentLineSearchProximalL1
import numpy as np
import matplotlib.pyplot as plt
import utils
import linear_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True)
    io_args = parser.parse_args()
    question = io_args.question


    if question == "2":
        data = utils.load_dataset("logisticData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        fun_obj = FunObjLogReg()
        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=400, verbose=True)
        model = linear_models.LogRegClassifier(fun_obj, optimizer)
        model.fit(X,y)

        print("LogReg Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("LogReg Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))
        print("# nonZeros: {:d}".format((model.w != 0).sum()))

    elif question == "2.1":
        data = utils.load_dataset("logisticData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        fun_obj = FunObjLogRegL2(1)
        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=400, verbose=False)
        model = linear_models.LogRegClassifier(fun_obj, optimizer)
        model.fit(X,y)

        print("\nlogRegL2 Training error %.3f" % utils.classification_error(model.predict(X), y))
        print("logRegL2 Validation error %.3f" % utils.classification_error(model.predict(X_valid), y_valid))
        print("# nonZeros: %d" % (model.w != 0).sum())

    elif question == "2.2":
        data = utils.load_dataset("logisticData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        """YOUR CODE HERE"""
        # Choosing best lambda
        lammys = np.array([1e-2, 1e-1, 1, 1e1])
        # best_lammy = None
        # best_err_valid = np.inf
        for lammy in lammys:
            fun_obj = FunObjLogReg()
            optimizer = OptimizerGradientDescentLineSearchProximalL1(lammy, fun_obj, X, y, max_evals=400, verbose=False)
            model = linear_models.LogRegClassifier(fun_obj, optimizer)
            model.fit(X,y)

            err_train = utils.classification_error(model.predict(X), y)
            err_valid = utils.classification_error(model.predict(X_valid), y_valid)
            n_steps = optimizer.num_evals
            n_nonzeros = np.sum(model.w != 0)

            print("lammy={:.3}, training error={:.3f}, validation error={:.3f}, #non-zeros={:d}, #gradient descent iterations={:d}".format(lammy, err_train, err_valid, n_nonzeros, n_steps))

    elif question == "2.3":
        data = utils.load_dataset("logisticData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        local_fun_obj = FunObjLogReg()
        optimizer = OptimizerGradientDescentLineSearch(local_fun_obj, X, y, max_evals=400, verbose=False)
        global_fun_obj = FunObjLogRegL0(1)
        model = linear_models.LogRegClassifierForwardSelection(global_fun_obj, optimizer)
        model.fit(X,y)

        print("LogReg Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("LogReg Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))
        print("# nonZeros: {:d}".format((model.w != 0).sum()))

    elif question == "3":
        data = utils.load_dataset("multiData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        model = linear_models.LeastSquaresClassifier()
        model.fit(X, y)

        print("LeastSquaresClassifier Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("LeastSquaresClassifier Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))

        print(np.unique(model.predict(X)))


    elif question == "3.2":
        data = utils.load_dataset("multiData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        fun_obj = FunObjLogReg()
        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=500, verbose=False)
        model = linear_models.LogRegClassifierOneVsAll(fun_obj, optimizer)
        model.fit(X, y)

        print("LogRegClassifierOneVsAll Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("LogRegClassifierOneVsAll Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))
        print(np.unique(model.predict(X))) 

    elif question == "3.4":
        data = utils.load_dataset("multiData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        fun_obj = FunObjSoftmax()
        optimizer = OptimizerGradientDescentLineSearch(fun_obj, X, y, max_evals=500, verbose=True)
        model = linear_models.MulticlassLogRegClassifier(fun_obj, optimizer)
        model.fit(X, y)

        print("Softmax Training error: {:.3f}".format(utils.classification_error(model.predict(X), y)))
        print("Softmax Validation error: {:.3f}".format(utils.classification_error(model.predict(X_valid), y_valid)))

    elif question == "3.5":
        data = utils.load_dataset("multiData")
        X, y = data['X'], data['y']
        X_valid, y_valid = data['Xvalid'], data['yvalid']

        """YOUR CODE HERE FOR Q3.5"""
        raise NotImplementedError()