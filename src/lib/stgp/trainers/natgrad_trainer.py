import jax
from jax.scipy.optimize import minimize
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten, tree_unflatten
import jax.numpy as jnp

import objax
from objax import ModuleList, TrainRef, TrainVar
import numpy as np
from typing import List, Union

from .trainer import Trainer, vc_remove_vars
from ..utils.utils import vc_keep_vars, match_suffix, get_parameters, get_var_name_with_id

from ..dispatch import _ensure_str

def _get_likelihood(lik):
    try:
        return lik.likelihood_arr[0]
    except Exception as e:
        return lik

# TODO: refactor this as it is repeapted inside the nat_grad files
def get_mean_field_approx_posterior_name_list(model, param_dict, approx_posterior):
    q_type = _ensure_str(approx_posterior.approx_posteriors[0])
    if q_type == 'MeanFieldAcrossDataApproximatePosterior':
        m_name_list = []
        S_chol_name_list = []
        for q in approx_posterior.approx_posteriors:
            _m_list, _S_list = get_mean_field_approx_posterior_name_list(model, param_dict, q)
            m_name_list = m_name_list+_m_list
            S_chol_name_list = S_chol_name_list+_S_list
    else:
        m_name_list = []
        S_chol_name_list = []
        for q in approx_posterior.approx_posteriors:
            m_name = get_var_name_with_id(model, id(q._m.raw_var), param_dict)
            S_chol_name = get_var_name_with_id(model, id(q._S_chol.raw_var), param_dict)
            m_name_list.append(m_name)
            S_chol_name_list.append(S_chol_name)

    return m_name_list, S_chol_name_list


def get_vars_to_update(model, vc):
    approx_posterior = model.approximate_posterior

    param_dict = get_parameters(model, replace_name=False, return_id=True, return_state_var=True)

    q_type = _ensure_str(approx_posterior)

    if q_type == 'MeanFieldApproximatePosterior':
        m_name_list, S_chol_name_list = get_mean_field_approx_posterior_name_list(model, param_dict, approx_posterior)
            
    elif q_type == 'FullGaussianApproximatePosterior':
        m_name = get_var_name_with_id(model, id(approx_posterior._m.raw_var), param_dict)
        S_chol_name = get_var_name_with_id(model, id(approx_posterior._S_chol.raw_var), param_dict)

        m_name_list = [m_name]
        S_chol_name_list = [S_chol_name]

    elif q_type == 'MeanFieldConjugateGaussian':
        # collect conjugate vars
        m_name_list = []
        S_chol_name_list = []


        for q in approx_posterior.approx_posteriors:
            lik_var = _get_likelihood(q.surrogate.likelihood).variance_param

            Y_name = get_var_name_with_id(model, id(q.surrogate.data._Y.raw_var), param_dict)
            V_chol_name = get_var_name_with_id(model, id(lik_var.raw_var), param_dict)

            m_name_list.append(Y_name)
            S_chol_name_list.append(V_chol_name)

    elif q_type in ['FullConjugateGaussian', 'FullConjugatePrecisionGaussian']:
        # precision gaussian simply extend the full conjugate gaussian
        # and so , confusingly, the precision is actually stored in variance_param

        q = approx_posterior

        lik_var = q.surrogate.likelihood.variance_param

        Y_name = get_var_name_with_id(model, id(q.surrogate.data._Y.raw_var), param_dict)
        V_chol_name = get_var_name_with_id(model, id(lik_var.raw_var), param_dict)

        m_name_list = [Y_name]
        S_chol_name_list = [V_chol_name]

    else:
        breakpoint()
        raise RuntimeError()

    return vc_keep_vars(vc, [*m_name_list, *S_chol_name_list])

def update_vars(model, vars_to_update, params):
    """
    Natural gradients are computed in the natural parameterisation.
    """
    approx_posterior = model.approximate_posterior

    q_type = _ensure_str(approx_posterior)

    if _ensure_str(approx_posterior) in ['MeanFieldApproximatePosterior']:
        q_arr = approx_posterior.approx_posteriors

        new_params = []
        for q in range(len(q_arr)):
            new_params += [params[0][q], params[1][q]]

        vars_to_update.assign(new_params)

    elif q_type == 'MeanFieldConjugateGaussian':
        q_arr = approx_posterior.approx_posteriors

        new_params = []
        for q in range(len(q_arr)):

            lik_var = _get_likelihood(q_arr[q].surrogate.likelihood).variance_param

            Y_shape = q_arr[q].surrogate.data._Y.raw_var.shape
            new_params += [np.reshape(params[0][q], Y_shape), lik_var.inv_transform(params[1][q])]

        vars_to_update.assign(new_params)

    elif q_type == 'FullGaussianApproximatePosterior':
        vars_to_update.assign(params)

    elif q_type in ['FullConjugateGaussian', 'FullConjugatePrecisionGaussian']:
        lik_var = approx_posterior.surrogate.likelihood.variance_param

        new_params = [
            params[0],
            lik_var.inv_transform(params[1])
        ]

        vars_to_update.assign(new_params)
    else:
        raise RuntimeError()

MOMENTUM_TERMS = []

class NatGradTrainer(Trainer):
    def __init__(
        self, 
        m,
        schedule = None,
        total_epochs = None,
        enforce_psd_type=None,
        prediction_samples=None,
        return_objective = True,
        nan_max_attempt = None
    ):
        """
        Args
            schedule: [linear, log, constant, none]
        """
        self.m = m
        vc = self.m.vars()

        self.vars_to_update = get_vars_to_update(self.m, vc)

        self.natgrad_fn = objax.Jit(
            self.m.natural_gradient_update,
            vc,
            static_argnums=[1, 2]
        )

        self.return_objective = return_objective
        if self.return_objective:
            self.objective_fn = objax.Jit(
                self.m.get_objective, 
                vc
            )
        else:
            self.objective_fn = None


        if (schedule is None) or (schedule == 'none'):
            schedule = 'constant'

        self.schedule = schedule

        # When using a schedule we need to know how many iters we are training for
        #    so we can construct the schedule
        self.total_epochs = total_epochs

        self.enforce_psd_type = enforce_psd_type

        self.prediction_samples = prediction_samples

        if nan_max_attempt is None:
            nan_max_attempt = 1

        self.nan_max_attempt = nan_max_attempt

    def train(
        self, 
        learning_rate, 
        epochs, 
        callback=None,
        epoch_ofset=0,
        verbose=False,
        raise_error = True,
        momentum=False,
        momentum_rate = 0.1
    ):
        global MOMENTUM_TERMS

        def gradient_step(i, global_i, momentum, momentum_terms):

            if self.total_epochs:
                percent = (global_i+1)/self.total_epochs
            else:
                percent = (i+1)/epochs

            if self.schedule == 'linear':
                lr = learning_rate[1] * percent + (1-percent) * learning_rate[0]

            elif self.schedule == 'log':
                lr = np.power(learning_rate[1], percent) * np.power(learning_rate[0], (1-percent))

            elif self.schedule == 'constant':
                if type(learning_rate) is not list:
                    lr = learning_rate
                else:
                    lr = learning_rate[0]
            else:
                raise NotImplementedError(f'{self.schedule} is not implemented')

            if verbose:
                print(f'{i} / {epochs} -- {global_i} / {self.total_epochs} -- {lr}')

            params = self.natgrad_fn(lr, self.enforce_psd_type, self.prediction_samples)

            if momentum:
                #print(f'using momentum -- {momentum_rate}')
                if len(momentum_terms) == 0:
                    momentum_terms = [params]
                elif len(momentum_terms) == 1:
                    # [new, old]
                    momentum_terms = [params, momentum_terms[0]]
                else:
                    new_param_1 = params[0] + momentum_rate * (momentum_terms[1][0] - momentum_terms[0][0])

                    # update in cholesky space to ensure PSDness
                    new_param_2_chol = np.linalg.cholesky(params[1]) + momentum_rate * (np.linalg.cholesky(momentum_terms[1][1]) - np.linalg.cholesky(momentum_terms[0][1]))
                    new_param_2 = new_param_2_chol @ np.transpose(new_param_2_chol, [0, 2, 1])

                    params = [new_param_1, new_param_2]
                    #params = [new_param_1, params[1]]

                    # [new, old]
                    momentum_terms = [params, momentum_terms[0]]

            if np.any(np.isnan(params[0])) or np.any(np.isnan(params[1])):
                print('NaN encountered whilst natgrad training!')
                return True, momentum_terms

            update_vars(self.m, self.vars_to_update, params)
            return False, momentum_terms

        epoch_arr = []
        momentum_terms = MOMENTUM_TERMS

        for i in range(epochs):
            max_attempt = self.nan_max_attempt

            while True:
                err_flag, momentum_terms = gradient_step(i, i+epoch_ofset, momentum, momentum_terms)

                if not err_flag:
                    break

                # we have run out of attempts
                if max_attempt == 0:
                    if raise_error:
                        raise RuntimeError('NaN encountered whilst natgrad training!')
                    else:
                        print('NaN encountered whilst natgrad training!')
                        return jnp.array(epoch_arr).flatten(), None

                print('retrying nat grad')
                max_attempt = max_attempt - 1

            if self.return_objective:
                val = self.objective_fn()
                epoch_arr.append(jnp.array(val).flatten())
            else:
                epoch_arr.append(np.nan)


            if callback is not None:
                callback(i, None, None)

        if momentum:
            MOMENTUM_TERMS = momentum_terms

        return jnp.array(epoch_arr).flatten(), None

