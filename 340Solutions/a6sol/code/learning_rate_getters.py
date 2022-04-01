import numpy as np

class LearningRateGetter:

    def __init__(self, multiplier):
        self.multiplier = multiplier
        self.num_evals = 0

    def reset(self):
        self.num_evals = 0

    def get_learning_rate(self):
        raise NotImplementedError        

class LearningRateGetterConstant(LearningRateGetter):
    
    def get_learning_rate(self):
        """YOUR CODE HERE FOR Q4.2"""
        # raise NotImplementedError()
        self.num_evals += 1
        return self.multiplier

class LearningRateGetterInverse(LearningRateGetter):
    
    def get_learning_rate(self):
        """YOUR CODE HERE FOR Q4.2"""
        # raise NotImplementedError()
        self.num_evals += 1  # should safeguard against divide-by-zero
        return self.multiplier / self.num_evals

class LearningRateGetterInverseSquared(LearningRateGetter):

    def get_learning_rate(self):
        """YOUR CODE HERE FOR Q4.2"""
        # raise NotImplementedError()
        self.num_evals += 1  # should safeguard against divide-by-zero
        return self.multiplier / (self.num_evals ** 2)

class LearningRateGetterInverseSqrt(LearningRateGetter):

    def get_learning_rate(self):
        """YOUR CODE HERE FOR Q4.2"""
        # raise NotImplementedError()
        self.num_evals += 1  # should safeguard against divide-by-zero
        return self.multiplier / np.sqrt(self.num_evals)
