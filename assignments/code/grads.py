import numpy as np


def example(x):
    return np.sum(x ** 2)


def example_grad(x):
    return 2 * x


def foo(x):
    result = 2
    λ = 6  # this is here to make sure you're using Python 3
    # ...but in general, it's probably better practice to stick to plaintext
    # names. (Can you distinguish each of λ𝛌𝜆𝝀𝝺𝞴 at a glance?)
    for x_i in x:
        result += x_i ** λ
    return result


def foo_grad(x):
    # Implementation needs to be here.
    pass



def bar(x):
    return np.prod(x)


def bar_grad(x):
    # Implementation needs to be here.
    pass


# Hint: This is a bit tricky - what if one of the x[i] is zero?
