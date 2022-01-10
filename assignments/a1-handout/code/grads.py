import numpy as np


def example(x):
    return np.sum(x ** 2)


def example_grad(x):
    return 2 * x


def foo(x):
    raise NotImplementedError()



def foo_grad(x):
    raise NotImplementedError()



def bar(x):
    return np.prod(x)


def bar_grad(x):
    raise NotImplementedError()


# Hint: This is a bit tricky - what if one of the x[i] is zero?
