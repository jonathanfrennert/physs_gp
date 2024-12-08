from ... import settings
from ..matrix_ops import cholesky, cholesky_solve, triangular_solve, add_jitter, lower_triangle, vectorized_lower_triangular_cholesky, vectorized_lower_triangular, lower_triangular_cholesky, lower_triangle, to_lower_triangular_vec, vectorized_cholesky_to_psd
from ...utils.utils import vc_keep_vars, get_parameters, get_var_name_with_id, get_batch_type
from ...utils.nan_utils import get_same_shape_mask
from ...dispatch import dispatch, evoke
from ...approximate_posteriors import ConjugateApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullConjugateGaussian, FullGaussianApproximatePosterior
from ..parameter_transforms import psd_retraction_map
from ..permutations import right_permute_mat, permute_mat
from ..integrals.approximators import mv_block_monte_carlo, mv_mean_field_block_monte_carlo
from ..matrix_ops import lower_triangle
from ...dispatch import _ensure_str

from ..elbos.elbos import compute_expected_log_liklihood
from .exponential_family_transforms import xi_to_theta, theta_to_lambda, xi_to_expectation, expectation_to_xi, lambda_to_theta, theta_to_xi, lambda_to_xi, xi_to_lambda, reparametise_cholesky_grad


import chex
from batchjax import batch_or_loop, BatchType
import jax
import jax.numpy as np
from jax import grad, jit, jacfwd, jacrev
from jax import vjp, jvp
from functools import partial

import objax

from typing import List


class RevJac(objax.gradient._DerivativeBase):
    """
    Copies objax implementation of Jacobian except it uses jacrev instead of jacfwd
         This is because for some reason jacrev seems more computational stable than jacfwd?
    """

    def __init__(self,
                 f,
                 variables,
                 input_argnums = None):
        """Constructs an instance to compute the Jacobian of f w.r.t. variables and arguments.
        Args:
            f: the function for which to compute Jacobian.
            variables: the variables for which to compute gradients.
            input_argnums: input indexes, if any, on which to compute gradients.
        """
        super().__init__(lambda f_func: jax.jacrev(f_func, has_aux=True),
                         f=f,
                         variables=variables,
                         input_argnums=input_argnums)

def _get_mean_field_approx_posterior_name_list(model, param_dict, approx_posterior):
    q_type = _ensure_str(approx_posterior.approx_posteriors[0])
    if q_type == 'MeanFieldAcrossDataApproximatePosterior':
        m_name_list = []
        S_chol_name_list = []
        for q in approx_posterior.approx_posteriors:
            _m_list, _S_list = _get_mean_field_approx_posterior_name_list(model, param_dict, q)
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


def _get_mf_params_names(model):
    """
    Returns a list of all the variational mean and variance names from a mean field approximate posterior
    """

    param_dict = get_parameters(model, replace_name=False, return_id=True)
    return _get_mean_field_approx_posterior_name_list(model, param_dict, model.approximate_posterior)

    m_name_list = []
    S_chol_list = []
    for q in approx_posteriors:

        m_name = get_var_name_with_id(model, id(q._m.raw_var), param_dict)
        S_chol_name = get_var_name_with_id(model, id(q._S_chol.raw_var), param_dict)

        m_name_list.append(m_name)
        S_chol_list.append(S_chol_name)

    return m_name_list, S_chol_list

def _get_fp_params_names(model):
    """
    Returns a the variational mean and variance name from a full-Gaussian approximate posterior
    """
    q = model.approximate_posterior
    param_dict = get_parameters(model, replace_name=False, return_id=True)


    m_name = get_var_name_with_id(model, id(q._m.raw_var), param_dict)
    S_chol_name = get_var_name_with_id(model, id(q._S_chol.raw_var), param_dict)

    return m_name, S_chol_name

