import numpy as np
from numpy.linalg import norm

"""
Implementation of optimizers, following the design pattern of PyTorch,
a popular library for differentiable programming and optimization.

Optimizers are used with function objects. See fun_obj.py.
"""

class Optimizer:

    def step(self):
        raise NotImplementedError()

    def set_parameters(self, parameters):
        self.parameters = parameters

    def set_fun_obj_args(self, *fun_obj_args):
        self.fun_obj_args = fun_obj_args

    def reset(self):
        """
        In case we want to re-run the optimization with different parameters, etc.
        """
        raise NotImplementedError()

class OptimizerGradientDescent(Optimizer):
    """
    Vanilla gradient descent algorithm, implemented into an Optimizer class
    """

    def __init__(self, fun_obj, *fun_obj_args, optimal_tolerance=1e-2, learning_rate=1e-3, max_evals=1000, verbose=False):
        self.parameters = None
        self.optimal_tolerance = optimal_tolerance
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate  # for resetting
        self.fun_obj = fun_obj
        self.fun_obj_args = fun_obj_args  # X and y
        self.max_evals = max_evals
        self.num_evals = 0
        self.verbose = verbose

        # Keep f and g as state variables to reduce redundancy
        self.f_old = None
        self.g_old = None

    def reset(self):
        """
        The state of the optimizer is tied to the state of the parameters.
        Resetting an optimizer will revert its state to the original.
        In practice, it doesn't matter whether you use reset() or initialize a new optimizer,
        but for A4 we will re-use the same optimizer, because it's more convenient and generic.
        """
        self.num_evals = 0
        self.parameters = None
        self.fun_obj_args = None
        self.learning_rate = self.initial_learning_rate
        self.f_old = None
        self.g_old = None

    def step(self):
        """
        step() does not have any argument because the parameters for optimization
        are registered via the constructor with the "parameters" argument.
        Calling optimizer.step() will conduct one step of gradient descent, i.e.
        w^{t+1} = w^t - \alpha^t * \nabla f(w^t)

        For A4, more refactoring is done to improve wall-clock time efficiency.
        """

        if self.parameters is None:
            raise RuntimeError("You must set the parameters for the optimizer with set_parameters() before calling step().")

        if self.fun_obj_args is None:
            raise RuntimeError("You must set the arguments for the function object with set_fun_obj_args() before calling step().")

        # Evaluate old value and gradient
        if self.f_old is None or self.g_old is None:
            self.f_old, self.g_old = self.get_function_value_and_gradient(self.parameters)

        # Perform a step: learning rate tuning and gradient descent in one call
        # This is to reduce the number of evaluations by re-using solutions from line search
        w_new, f_new, g_new = self.get_learning_rate_and_step(self.f_old, self.g_old)
        self.parameters = w_new

        # Update optimizer state for faster compute
        self.f_old = f_new
        self.g_old = g_new

        self.num_evals += 1
        break_yes = self.break_yes(g_new)
        return f_new, g_new, self.parameters, break_yes

    def get_learning_rate_and_step(self, f_old, g_old):
        """
        For vanilla gradient descent, combining learning rate and step doesn't
        necessarily give us speedup, but for backtracking line search, we can cut down
        at least one gradient computation by returning the last-used f and g values during backtracking
        """
        w_old = self.parameters
        alpha = self.learning_rate
        w_new = w_old - alpha * g_old
        f_new, g_new = self.get_function_value_and_gradient(w_new)
        return w_new, f_new, g_new

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

    def get_next_parameter_value(self, alpha, g):
        """
        Get the new parameter value after the gradient descent step.
        Does not mutate self.parameters. step() will call this and then
        overwrite the values explicitly.
        """
        return self.parameters - alpha * g

    def get_function_value_and_gradient(self, w):
        """
        Evaluate function and gradient based on the input w.
        w is not necessarily the current parameter value.
        For vanilla gradient descent and line search, this is simply pass-through.
        For proximal and more advanced gradient methods, additional terms are introduced.
        """
        return self.fun_obj.evaluate(w, *self.fun_obj_args)

class OptimizerGradientDescentLineSearch(OptimizerGradientDescent):
    """
    You *don't* need to understand this code.
    An advanced version of gradient descent, using backtracking line search 
    to automate finding a good step size. Take CPSC 406 for more information!
    """

    def __init__(self, fun_obj, *fun_obj_args, optimal_tolerance=1e-2, gamma=1e-4, max_evals=1000, verbose=False):
        super().__init__(fun_obj, *fun_obj_args, optimal_tolerance=optimal_tolerance, learning_rate=None, max_evals=max_evals, verbose=verbose)
        self.gamma = gamma
        self.initial_alpha = 1.

    def get_learning_rate_and_step(self, f_old, g_old):
        # Quadratic interpolation requires squared norm of gradient
        gg = g_old@g_old
        w_old = self.parameters
        alpha = self.initial_alpha

        # For Wolfe condition, which requires "first estimate" results
        gtd = None  # to be populated within inner loop

        # Backtracking line search loop
        while True:
            # Peek: take one step
            w_new = self.get_next_parameter_value(alpha, g_old)
            f_new, g_new = self.get_function_value_and_gradient(w_new)

            if gtd is None:
                gtd = g_old @ (w_new - w_old)

            # End backtracking if my height is sufficiently low
            if self.backtracking_break_yes(f_new, f_old, alpha, gg, gtd):
                break

            if self.verbose:
                print("f_new: {:.3f} - f_old: {:.3f} - Backtracking...".format(f_new, f_old))

            # Mutate alpha according to Armijo-Goldstein
            alpha = self.get_backtracked_alpha(f_new, f_old, alpha, gg, gtd)

        # "Good alpha" will carry over.
        self.initial_alpha = self.get_good_next_alpha(alpha, g_new, g_old)

        # Safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.

        return w_new, f_new, g_new

    def get_good_next_alpha(self, alpha, g_new, g_old):
        """
        Carry over the good alpha value
        """
        y = g_new - g_old
        alpha = -alpha * (y@g_old) / (y@y)
        # Safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.
        return alpha

    def get_backtracked_alpha(self, f_new, f_old, alpha, *multiplier_ingredients):
        """
        Our line search implementation reduces step size based on gradient's L2 norm
        Proximal gradient method just cuts it in half.
        """
        gg, gtd = multiplier_ingredients
        left = f_new - f_old
        right = alpha * gg
        return (alpha ** 2) * gg/(2*(left + right))

    def backtracking_break_yes(self, f_new, f_old, alpha, *multiplier_ingredients):
        """
        Our default Armijo search uses gradient's squared L2 norm as multiplier.
        Proximal gradient will use dot product between 
        gradient g and parameter displacement (w_new - w_old) as multiplier.
        """
        gg, gtd = multiplier_ingredients
        return f_new <= f_old - self.gamma * alpha * gg

class OptimizerGradientDescentLineSearchProximalL1(OptimizerGradientDescentLineSearch):
    """
    You *don't* need to understand this code.
    An implementation of proximal gradient method for enabling L1 regularization.
    The input function object should be just the desired loss term *without penalty*.

    NOTE: my apologies for the long class names, but I prefer clarity over brevity ;)
    (also most people use double-clicking, ctrl-c, ctrl-v for short names too anyways)
    """

    def __init__(self, lammy, fun_obj, *fun_obj_args, optimal_tolerance=1e-2, gamma=1e-4, max_evals=1000, verbose=False):
        """
        Note that lammy is passed to the optimizer, not the function object.
        """
        super().__init__(fun_obj, *fun_obj_args, optimal_tolerance=optimal_tolerance, gamma=gamma, max_evals=max_evals, verbose=verbose)
        self.lammy = lammy

    def get_backtracked_alpha(self, f_new, f_old, alpha, *multiplier_ingredients):
        """
        Proximal gradient method just cuts it in half.
        """
        return alpha / 2.

    def get_next_parameter_value(self, alpha, g):
        """
        For proximal gradient for L1 regularization, first make a vanilla GD step,
        and then apply proximal operator.
        """
        w_new = super().get_next_parameter_value(alpha, g)
        w_proxed = self._get_prox_l1(w_new, alpha)
        return w_proxed

    def backtracking_break_yes(self, f_new, f_old, alpha, *multiplier_ingredients):
        """
        Our default Armijo search uses gradient's squared L2 norm as multiplier.
        Proximal gradient will use Wolfe condition. Use dot product between 
        gradient g and parameter displacement (w_new - w_old) as multiplier.
        f_new and f_old already incorporate L1 regularization.
        """
        gg, gtd = multiplier_ingredients
        return f_new <= f_old - self.gamma * alpha * gtd

    def get_function_value_and_gradient(self, w):
        """
        Evaluate f and then add the L1 regularization term.
        Don't mutate g here.
        """
        f, g = super().get_function_value_and_gradient(w)
        f += self.lammy * np.sum(np.abs(w))
        return f, g

    def break_yes(self, g):
        w = self.parameters
        optimal_condition = norm(w - self._get_prox_l1(w - g, 1.0), float('inf'))
        if optimal_condition < self.optimal_tolerance:
            if self.verbose:
                print("Problem solved up to optimality tolerance {:.3f}".format(self.optimal_tolerance))
            return True
        elif self.num_evals >= self.max_evals:
            if self.verbose:
                print("Reached maximum number of function evaluations {:.3f}".format(self.max_evals))
            return True
        else:
            return False

    def _get_prox_l1(self, w, alpha):
        return np.sign(w) * np.maximum(abs(w) - self.lammy * alpha, 0)

