from numpy.linalg import norm

"""
Implementation of optimizers, following the design pattern of PyTorch,
a popular library for differentiable programming and optimization.

Optimizers are used with function objects. See fun_obj.py.
"""

class Optimizer:

    def step(self):
        raise NotImplementedError()

    def get_learning_rate(self, f_old, g_old):
        raise NotImplementedError()

    def set_parameters(self, parameters):
        self.parameters = parameters

class OptimizerGradientDescent(Optimizer):
    """
    Vanilla gradient descent algorithm, implemented into an Optimizer class
    """

    def __init__(self, fun_obj, *fun_obj_args, optimal_tolerance=1e-2, learning_rate=1e-3, max_evals=1000, verbose=False):
        self.parameters = None
        self.optimal_tolerance = optimal_tolerance
        self.learning_rate = learning_rate
        self.fun_obj = fun_obj
        self.fun_obj_args = fun_obj_args  # X and y
        self.max_evals = max_evals
        self.num_evals = 0
        self.verbose = verbose

    def step(self):
        """
        step() does not have any argument because the parameters for optimization
        are registered via the constructor with the "parameters" argument.
        Calling optimizer.step() will conduct one step of gradient descent, i.e.
        w^{t+1} = w^t - \alpha^t * \nabla f(w^t)

        PUT ANSWERS TO THESE QUESTIONS IN YOUR PDF:        
        Q1: What's \alpha^t in my code?
        Q2: What's \nabla f(w^t) in my code?
        Q3: What's \w^t in my code?
        Q4: What's break_yes doing?
        """

        if self.parameters is None:
            raise RuntimeError("You must set the parameters for the optimizer with set_parameters() before calling step().")

        # Evaluate old value and gradient
        f_old, g_old = self.fun_obj.evaluate(self.parameters, *self.fun_obj_args)

        # Determine step size (no-op for vanilla gradient descent)
        self.learning_rate = self.get_learning_rate(f_old, g_old)

        # Perform a step
        self.parameters = self.parameters - self.learning_rate * g_old

        # Evaluate new value and gradient
        f_new, g_new = self.fun_obj.evaluate(self.parameters, *self.fun_obj_args)

        self.num_evals += 1
        break_yes = self.break_yes(g_new)
        return f_new, g_new, self.parameters, break_yes

    def get_learning_rate(self, f_old, g_old):
        # For vanilla gradient descent, don't do anything.
        # For backtracking line search, use f value and gradient.
        return self.learning_rate        

    def break_yes(self, g):
        gradient_norm = norm(g, float('inf'))
        if gradient_norm < self.optimal_tolerance:
            if self.verbose:
                print("Problem solved up to optimality tolerance {:.3f}".format(self.optimal_tolerance))
            return True
        elif self.num_evals >= self.max_evals:
            if self.verbose:
                print("Reached maximum number of function evaluations {:.3f}".format(self.max_evals))
            return True
        else:
            return False

class OptimizerGradientDescentLineSearch(OptimizerGradientDescent):

    """
    You *don't* need to understand this code.
    An advanced version of gradient descent, using backtracking line search 
    to automate finding a good step size. Take CPSC 406 for more information!
    """

    def __init__(self, fun_obj, *fun_obj_args, optimal_tolerance=1e-2, gamma=1e-4, max_evals=1000, verbose=False):
        super().__init__(fun_obj, *fun_obj_args, optimal_tolerance=optimal_tolerance, learning_rate=None, max_evals=max_evals, verbose=verbose)
        self.gamma = gamma

    def get_learning_rate(self, f_old, g_old):
        """
        Backtracking line search to tune step size.
        """
        # Quadratic interpolation requires squared norm of gradient
        gg = g_old.T@g_old
        alpha = 1.
        # Backtracking line search loop
        while True:
            # Peek: take one step
            w_new = self.parameters - alpha * g_old
            f_new, g_new = self.fun_obj.evaluate(w_new, *self.fun_obj_args)

            # End backtracking if my height is sufficiently low
            if f_new <= f_old - self.gamma * alpha * gg:
                break

            if self.verbose:
                print("f_new: {:.3f} - f_old: {:.3f} - Backtracking...".format(f_new[0], f_old[0]))

            # Mutate alpha according to Armijo-Goldstein
            left = f_new - f_old
            right = alpha * gg
            alpha = alpha ** 2 * gg/(2*(left + right))

        self.learning_rate = alpha
        return self.learning_rate

