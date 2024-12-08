import jax
from scipy.optimize import minimize
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp

import objax
from objax import ModuleList, TrainRef, TrainVar
import numpy as np

from timeit import default_timer as timer

import json
import typing
from typing import List, Union

from ..utils.utils import vc_remove_vars, vc_keep_vars

import inspect
from typing import List, Optional, Callable, Tuple, Dict, Union

import jax

from objax.module import Function, Module
from objax.typing import JaxArray
from objax.util import repr_function, class_name
from objax.variable import BaseState, TrainVar, VarCollection
from objax.gradient import _DerivativeBase

class ReverseModeGrad(_DerivativeBase):
    """The Grad module is used to compute the gradients of a function."""

    def __init__(self, f: Callable,
                 variables: Optional[VarCollection],
                 input_argnums: Optional[Tuple[int, ...]] = None):
        """Constructs an instance to compute the gradient of f w.r.t. variables.

        Args:
            f: the function for which to compute gradients.
            variables: the variables for which to compute gradients.
            input_argnums: input indexes, if any, on which to compute gradients.
        """
        super().__init__(lambda f_func: jax.jacrev(f_func, has_aux=True),
                         f=f,
                         variables=variables,
                         input_argnums=input_argnums)
        signature = inspect.signature(f)
        self.__wrapped__ = f
        self.__signature__ = signature.replace(return_annotation=List[JaxArray])

    def __call__(self, *args, **kwargs):
        """Returns the computed gradients for the first value returned by `f`.

        Returns:
            A list of input gradients, if any, followed by the variable gradients."""
        return super().__call__(*args, **kwargs)

class ForwardModeGrad(_DerivativeBase):
    """The Grad module is used to compute the gradients of a function."""

    def __init__(self, f: Callable,
                 variables: Optional[VarCollection],
                 input_argnums: Optional[Tuple[int, ...]] = None):
        """Constructs an instance to compute the gradient of f w.r.t. variables.

        Args:
            f: the function for which to compute gradients.
            variables: the variables for which to compute gradients.
            input_argnums: input indexes, if any, on which to compute gradients.
        """
        super().__init__(lambda f_func: jax.jacfwd(f_func, has_aux=True),
                         f=f,
                         variables=variables,
                         input_argnums=input_argnums)
        signature = inspect.signature(f)
        self.__wrapped__ = f
        self.__signature__ = signature.replace(return_annotation=List[JaxArray])

    def __call__(self, *args, **kwargs):
        """Returns the computed gradients for the first value returned by `f`.

        Returns:
            A list of input gradients, if any, followed by the variable gradients."""
        return super().__call__(*args, **kwargs)


class Trainer:
    """
    All trainers are initalised with:
        m: model object
        optimizer: [list[str], str]
        opt_args: [none, dict] 
        hold_vars: list of vars to not train

    In addition to hold_vars, if there any parameters that been held they will also not be trained

    This is so all the required functions can be jitted on initialisation, and then the trainer object
       can be reused without further jitting

    """

    def get_all_hold_vars(self, hold_vars):
        if hold_vars is None:
            hold_vars = []

        # Get variables that have train=False
        hold_vars += self.m.get_fixed_params()

        return hold_vars

    def __init__(self, m, optimizer, opt_args = None, hold_vars = None, forward_mode = False):
        if opt_args == None:
            opt_args = {}

        self.m = m

        # Collect variables to train
        all_vars = m.vars()

        hold_vars = self.get_all_hold_vars(hold_vars)

        if len(hold_vars) > 0:
            vars_to_train = vc_remove_vars(all_vars, hold_vars)
        else:
            vars_to_train = all_vars

        # Jit required functions
        objective_fn = objax.Jit(self.m.get_objective, all_vars)

        if forward_mode:
            self.grad_fn = objax.Jit(
                ForwardModeGrad(objective_fn, vars_to_train), 
                all_vars
            )
        else:
            self.grad_fn = objax.Jit(ReverseModeGrad(objective_fn, vars_to_train), all_vars)


        self.objective_fn = objective_fn

        # Get optimizer
        self.opt = optimizer(vars_to_train, **opt_args)
        self.vars_to_train = vars_to_train
        self.all_vars = all_vars

class ScipyTrainer(Trainer):
    """
    A simple wrapper around scipy optimizers.

    Example:

    learning_curve, training_time = ScipyTrainer().train(
        m, 
        'BFGS',
        0.01,
        epochs,
        callback = None

    Heavily based on https://gist.github.com/slinderman/24552af1bdbb6cb033bfea9b2dc4ecfd with modifications to work with objax
    """
    def __init__(self, m, optimizer, opt_args = None, hold_vars = None):

        if opt_args == None:
            opt_args = {}

        self.m = m
        self.optimizer = optimizer

        # Collect variables to train
        all_vars = self.m.vars()

        hold_vars = self.get_all_hold_vars(hold_vars)

        m_vc = m.vars()

        # Only keep the trainable vars without the hold vars
        trainable_vc = m_vc.subset(TrainVar)

        if len(hold_vars) > 0:
            trainable_vc = vc_remove_vars(trainable_vc, hold_vars)
        else:
            trainable_vc = trainable_vc

        self.trainable_vc = trainable_vc


        # Jit required functions
        objective_fn = objax.Jit(self.m.get_objective, all_vars)

        self.grad_fn = objax.Jit(
            objax.Grad(objective_fn, self.trainable_vc), 
            all_vars
        )


        self.objective_fn = objective_fn

    def train(self, learning_rate, epochs, callback=None, ng_trainer = False, ng_lr = None, raise_error=True, epoch_ofset = None):
        """ For consistency we accept learning_rate here although it is not used. """

        x0 = self.trainable_vc.tensors()
        x0_flat, unravel = ravel_pytree(x0)

        def fun_flat(x_flat):
            self.trainable_vc.assign(unravel(x_flat))

            if ng_trainer:
                # In variational models with natural gradients the ELL term will not change
                # until q(f) has been updated, therefore we take a natural gradient step to incorporate the new x_flat. 
                # TODO: this seems to be quite unstable, cant use ng_lr = 1?
                obj, _ = ng_trainer.train(ng_lr, 1)
                return np.squeeze(obj)

            return self.objective_fn()

        def grad_flat(x_flat):
            # Convert from flat to pytree and assign
            self.trainable_vc.assign(unravel(x_flat))

            # evaluate gradient
            g_flat, _ = ravel_pytree(self.grad_fn())

            return np.array(g_flat)

        learning_rates = []

        # Wrap the callback to consume a pytree
        def callback_wrapper(x_flat, *args):
            learning_rates.append(fun_flat(x_flat))

            if callback is not None:
                callback(None, None, None)

        results = minimize(
            fun_flat, 
            x0_flat, 
            method=self.optimizer, 
            jac = grad_flat,
            callback = callback_wrapper,
            options = {
                'disp': False,
                'maxiter': epochs,
            }
        )

        res_x = unravel(results.x)
        self.trainable_vc.assign(res_x)

        return jnp.array(learning_rates).flatten(), 0


class GradDescentTrainer(Trainer):
    def train(
        self,
        learning_rate,
        epochs,
        callback=None,
        epoch_ofset = None,
        raise_error = True
    ):
        start = timer()
        epoch_arr = []

        def train_op():
            grad = self.grad_fn()
            val = self.objective_fn()
            self.opt(learning_rate, grad)
            return grad, val

        for i in range(epochs):
            grad, val = train_op()

            if np.isnan(val):
                if raise_error:
                    raise RuntimeError('NaN encountered whilst training!')
                else:
                    print('NaN encountered whilst training!')
                    return jnp.array(epoch_arr).flatten(), None

            if callback is not None:
                callback(i, grad, val)

            # Clean up val
            epoch_arr.append(jnp.array(val).flatten())

        end = timer()
        training_time = end - start

        return jnp.array(epoch_arr).flatten(), training_time

class SwitchTrainer(Trainer):
    """
    For use when multiple trainers are used per training epoch.

    Example:

        # Only train approximate posterior through natural gradients
        for q in m.approximate_posterior.approx_posteriors:
            q._m.fix()
            q._S_chol.fix()

        grad_step = GradDescentTrainer(m, objax.optimizer.Adam)
        nat_grad_step = NatGradTrainer(m)

        trainer = SwitchTrainer(
            [grad_step, nat_grad_step],

        )
        trainer.train(
            100,
            [0.01, 1.0],
            [1, 1],
            None
        )

    We pass through the trainers grad_step, and nat_grad_step through the init function to minimize jitting.

    """
    def __init__(self, trainer_list: list, callback_idx = 0):
        """
        Args:
            callback_idx: when using a callback we need to decide which trainer we want to use. This is not always just the last one as sometimes this might not return a value (like when using natural gradients)]
        """
        self.trainer_list = trainer_list
        self.callback_idx = callback_idx

    def train(
        self,
        learning_rates: list,
        epochs: list,
        callback = None,
        raise_error = True,
        trainer_kwargs = None
    ):
        iters = epochs[1]
        epochs = int(epochs[0])

        start = timer()

        num_trainers = len(self.trainer_list)

        total_elbos = []
        completed_epochs = [0 for j in range(num_trainers)]

        try:
            for i in range(epochs):
                for j in range(num_trainers):
                    if trainer_kwargs is None:
                        trainer_j_kwargs = {}
                    else:
                        trainer_j_kwargs = trainer_kwargs[j]

                    lc_j, _ = self.trainer_list[j].train(
                        learning_rates[j], 
                        iters[j], 
                        None, # We do not support individual trainer callbacks
                        epoch_ofset = completed_epochs[j],
                        raise_error = True,
                        **trainer_j_kwargs
                    )

                    total_elbos.append(lc_j)

                    completed_epochs[j] += iters[j]

                # After calling all individual trainers we have completed one training epoch
                if callback is not None:
                    total_elbo_idx = -(num_trainers-self.callback_idx)
                    val_to_pass = np.array([total_elbos[total_elbo_idx]]).flatten()
                    if len(val_to_pass) > 0:
                        # we use flatten and [-1] to handle both scalars and arrays
                        val_to_pass = val_to_pass[-1]

                    callback(i, None, val_to_pass)

        except RuntimeError as e:
            # it is likely that a nan was encounted
            print('finishing early! probably due to nans')

        end = timer()
        training_time = end - start

        return total_elbos, training_time
