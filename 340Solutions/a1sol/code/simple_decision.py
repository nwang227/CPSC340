def predict(X):
    if X[0] > -80.305106:
        if X[1] > 36.453576: # this "if" statement is optional
            y = 0
        else:
            y = 0
    else:
        if X[1] > 37.669007:
            y = 0
        else:
            y = 1
    return y
