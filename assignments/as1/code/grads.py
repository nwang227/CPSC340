from cmath import inf
from math import prod
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
    return 6* x**5
    pass



def bar(x):
    return np.prod(x)


def bar_grad(x):
    if np.prod(x) != 0:
        return np.prod(x) * x**-1
    else:
        if x.size - np.count_nonzero(x) >= 2:
            return np.zeros(x.size)
        else:
            prod = 1
            for x_i in x:
                if x_i != 0:
                    prod *= x_i
            result = np.where(x == 0, prod, 0)
            return result
        
    pass



# Hint: This is a bit tricky - what if one of the x[i] is zero?
