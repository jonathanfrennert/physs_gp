import jax
import jaxopt
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten, tree_unflatten

from . import Trainer, ScipyTrainer
import numpy as np

import jax.numpy as jnp

from timeit import default_timer as timer

class JaxoptTrainer(ScipyTrainer):
    """
    A simple wrapper around Jaxopt optimizers.

    Example:

    learning_curve, training_time = ScipyTrainer().train(
        m, 
        'BFGS',
        0.01,
        epochs,
        callback = None

    Heavily based on https://gist.github.com/slinderman/24552af1bdbb6cb033bfea9b2dc4ecfd with modifications to work with objax
    """
    def __init__(self, m, optimizer, opt_args = None, hold_vars = None, maxiter=1):

        super(JaxoptTrainer, self).__init__(m, optimizer, opt_args = opt_args, hold_vars = hold_vars)

        x0 = self.trainable_vc.tensors()
        x0_flat, unravel = ravel_pytree(x0)

        self.original_m_vars_tensors = self.m.vars().tensors()


        def fun_flat(x_flat):
            self.m.vars().assign(self.original_m_vars_tensors)
            self.trainable_vc.assign(unravel(x_flat))
            val= self.objective_fn()
            return val

        def grad_flat(x_flat):
            self.m.vars().assign(self.original_m_vars_tensors)
            # Convert from flat to pytree and assign
            self.trainable_vc.assign(unravel(x_flat))

            # evaluate gradient
            g_flat, _ = ravel_pytree(self.grad_fn())

            return jnp.array(g_flat)

        self.flat_fun = lambda x_flat: (fun_flat(x_flat), grad_flat(x_flat))

        self.jax_opt_fn = self.optimizer(fun=self.flat_fun, maxiter=maxiter, value_and_grad=True)

    def train(self, learning_rate, epochs, callback=None, raise_error=True, epoch_ofset = None, state_has_obj = True):
        """ For consistency we accept learning_rate here although it is not used. """

        start = timer()

        x0, unravel  = ravel_pytree(self.trainable_vc.tensors())

        state = self.jax_opt_fn.init_state(x0)
        zero_step = self.jax_opt_fn._make_zero_step(x0, state)
        opt_step = self.jax_opt_fn.update(x0, state)

        breakpoint()

        epoch_arr = []

        # first step manually to get state
        xnew, state = self.jax_opt_fn.run(x0)
        res_x = unravel(xnew)

        # hack for now to reset the tracers that jaxopt is somehow introducing
        self.m.vars().assign(self.original_m_vars_tensors)
        self.trainable_vc.assign(res_x)
        if state_has_obj:
            val = state.value
        else:
            val = self.m.get_objective()

        epoch_arr.append(np.array(val))

        if callback is not None:
            callback(0, state, val)

        if epochs != 1:
            for i in range(epochs-1):
                # gradient step
                xnew, state = self.jax_opt_fn.update(xnew, state)
                # update parameters
                res_x = unravel(xnew)
                self.m.vars().assign(self.original_m_vars_tensors)
                self.trainable_vc.assign(res_x)

                # get_current_fn_val
                if state_has_obj:
                    val = state.value
                else:
                    val = self.m.get_objective()
                epoch_arr.append(np.array(val))
                #val = 0

                if callback is not None:
                    callback(i, state, val)


        end = timer()
        training_time = end - start

        return jnp.array(epoch_arr).flatten(), training_time


