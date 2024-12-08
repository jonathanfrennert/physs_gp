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

from ..elbos.elbos import compute_expected_log_liklihood
from .exponential_family_transforms import xi_to_theta, theta_to_lambda, xi_to_expectation, expectation_to_xi, lambda_to_theta, theta_to_xi, lambda_to_xi, xi_to_lambda, reparametise_cholesky_grad

from .nat_grad_utils import RevJac, _get_mf_params_names, _get_fp_params_names

import chex
from batchjax import batch_or_loop, BatchType
import jax
import jax.numpy as np
from jax import grad, jit, jacfwd, jacrev
from jax import vjp, jvp
from functools import partial

import objax

from typing import List

def laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples):
    approx_posterior = model.approximate_posterior

    m = approx_posterior.m
    S_chol = approx_posterior.S_chol
    M = m.shape[0]
    P = model.prior.output_dim
    Z = model.prior.base_prior.get_Z_blocks()
    Q = model.prior.base_prior.input_dim

    def kl():
        # Compute KL term
        KL = evoke('kullback_leibler', model.approximate_posterior, model.prior, whiten=model.inference.whiten)(
            model.approximate_posterior, model.prior, model.inference.whiten
        )

        return KL

    m_name, S_chol_name = _get_fp_params_names(model)
    f_vars_to_diff = vc_keep_vars(model.vars(), [m_name])
    S_vars_to_diff = vc_keep_vars(model.vars(), [S_chol_name])

    XS = model.data.X
    N = XS.shape[0]
    def likelihood_conditional_mean(X):
        f = model.predict_f(X, squeeze=False, fix_shapes=False, num_samples=prediction_samples, posterior=True)[0]

        f = np.squeeze(f)
        #flatten output
        stacked_mean =  np.vstack(model.likelihood.conditional_mean(f))

        stacked_Y = np.reshape(model.data.Y, [-1], order='F')
        # we want zeros where Y is nan so we can mask stacked_mean
        nan_mask = get_same_shape_mask(stacked_Y)[:, None]

        chex.assert_equal_shape([stacked_mean, nan_mask])
        return stacked_mean * nan_mask

    def likelihood_conditional_var(X):
        f = model.predict_f(X, squeeze=False, fix_shapes=False, num_samples=prediction_samples, posterior=True)[0]
        f = np.squeeze(f)
        stacked_covar =  np.vstack(model.likelihood.conditional_var(f))
        return stacked_covar 

    def compute_kl_grad(S_chol, K):
        # TODO: add checks to see when this is allowed
        if model.inference.whiten:
            S_inv = cholesky_solve(S_chol, np.eye(M))
            kl_partial_s =  -(0.5*S_inv - 0.5*np.eye(M))
        else:
            # closed form KL derivative
            S_inv = cholesky_solve(S_chol, np.eye(M))
            K_chol = cholesky(add_jitter(K, settings.jitter))
            K_inv = cholesky_solve(K_chol, np.eye(M))
            kl_partial_s =  -(0.5*S_inv - 0.5*K_inv)

        return kl_partial_s

    KL_grad_arr = compute_kl_grad(S_chol,  model.prior.base_prior.b_covar(Z, Z))

    #pred_fn_grad = objax.Jacobian(likelihood_conditional_mean, f_vars_to_diff)
    pred_fn_grad = RevJac(likelihood_conditional_mean, f_vars_to_diff)

    def compute_lambda(conditional_mean_grad, conditional_var, kl_partial_s):
        #conditional_mean_grad = conditional_mean_grad[:, 0, 0, :, 0]
        conditional_var = np.squeeze(1/conditional_var)[:, None]

        stacked_Y = np.reshape(model.data.Y, [-1], order='F')
        # we want zeros where Y is nan so we can mask stacked_mean
        nan_mask = get_same_shape_mask(stacked_Y)[:, None]
        chex.assert_equal_shape([conditional_var, nan_mask])
        conditional_var = conditional_var * nan_mask

        # compute np.diag(conditional_var) @ conditional_mean_grad efficiently
        lambda_2 =  -((-0.5 * conditional_mean_grad.T  @ (conditional_var * conditional_mean_grad)) - ( kl_partial_s))
        return lambda_2

    conditional_var = likelihood_conditional_var(model.data.X)
    J = pred_fn_grad(XS)[0][:, 0, :, 0]

    lambda_2 =  compute_lambda(J, np.hstack(conditional_var), KL_grad_arr)

    #objax.Grad(model.get_objective, f_vars_to_diff)()

    return lambda_2

def gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples):
    if prediction_samples is None:
        raise RuntimeError()
    
    laplace_samples = 1

    m_name, S_chol_name = _get_fp_params_names(model)
    f_vars_to_diff = vc_keep_vars(model.vars(), [m_name])
    S_vars_to_diff = vc_keep_vars(model.vars(), [S_chol_name])

    # keep orignal values 
    f_values = f_vars_to_diff.tensors()
    S_chol_values = S_vars_to_diff.tensors()

    M = f_values[0].shape[0]

    f_val = f_values[0][None, ...]
    S_chol_val = lower_triangle(S_chol_values[0], M)
    var_val = S_chol_val @ S_chol_val.T
    var_val = var_val[None, None, ...]
    
    #sample 
    def wrapped_fn(f):
        f_vars_to_diff.assign([f[0]])

        return laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, laplace_samples)


    lam = mv_block_monte_carlo(wrapped_fn, f_val, var_val, generator = model.inference.generator, num_samples = prediction_samples)

    # assign back original values
    f_vars_to_diff.assign(f_values)

    return lam


def laplace_gauss_newton_natural_gradient_for_meanfield_approx_posterior(model, beta, prediction_samples):
    approx_posterior = model.approximate_posterior
    Q = len(approx_posterior.approx_posteriors)

    m = approx_posterior.m
    S_chol = approx_posterior.S_chol
    M = m.shape[1]
    Z = model.prior.base_prior.get_Z_blocks()

    def kl():
        # Compute KL term
        KL = evoke('kullback_leibler', model.approximate_posterior, model.prior, whiten=model.inference.whiten)(
            model.approximate_posterior, model.prior, model.inference.whiten
        )

        return KL

    m_name_list, S_chol_list = _get_mf_params_names(model)
    f_vars_to_diff = vc_keep_vars(model.vars(), [*m_name_list])
    S_vars_to_diff = vc_keep_vars(model.vars(), [*S_chol_list])

    XS = model.data.X
    def likelihood_conditional_mean(X):
        # NxQxB
        f = model.predict_f(X, squeeze=False, diagonal=True, num_samples=prediction_samples, posterior=True)[0]

        # Q x N x 1 x B
        cond_f =  model.likelihood.conditional_mean(f)
        # TODO: fix for product likelihood...
        cond_f = cond_f[:, :, 0, :]

        nan_mask = (get_same_shape_mask(model.data.Y).T)[..., None]

        chex.assert_equal_shape([cond_f, nan_mask])

        return cond_f * nan_mask

    def likelihood_conditional_var(X):
        f = model.predict_f(X, squeeze=False, diagonal=True, num_samples=prediction_samples, posterior=True)[0]
        stacked_covar =  np.vstack(model.likelihood.conditional_var(f))
        return stacked_covar 


    def compute_kl_grad(S_chol, K):

        # TODO: add checks to see when this is allowed
        if model.inference.whiten:
            S_inv = cholesky_solve(S_chol, np.eye(M))
            kl_partial_s =  -(0.5*S_inv - 0.5*np.eye(M))
        else:
            # closed form KL derivative
            S_inv = cholesky_solve(S_chol, np.eye(M))
            K_chol = cholesky(add_jitter(K, settings.jitter))
            K_inv = cholesky_solve(K_chol, np.eye(M))
            kl_partial_s =  -(0.5*S_inv - 0.5*K_inv)

        return kl_partial_s

    K_arr = model.prior.base_prior.b_covar_blocks(Z, Z)
    KL_grad_arr = batch_or_loop(
        compute_kl_grad,
        [S_chol,  K_arr],
        [0, 0],
        dim = Q,
        out_dim=1,
        batch_type = get_batch_type(approx_posterior.approx_posteriors)
    )

    #pred_fn_grad = objax.Jacobian(likelihood_conditional_mean, f_vars_to_diff)
    pred_fn_grad = RevJac(likelihood_conditional_mean, f_vars_to_diff)
    J = pred_fn_grad(model.data.X)


    def compute_lambda(conditional_mean_grad, conditional_var, kl_partial_s):
        conditional_mean_grad = np.reshape(conditional_mean_grad[:, :, 0, :, 0], [conditional_var.shape[0], -1])
        conditional_var = np.squeeze(1/conditional_var)[:, None]

        stacked_Y = np.reshape(model.data.Y, [-1], order='F')
        # we want zeros where Y is nan so we can mask stacked_mean
        nan_mask = get_same_shape_mask(stacked_Y)[:, None]
        chex.assert_equal_shape([conditional_var, nan_mask])
        conditional_var = conditional_var * nan_mask


        # compute np.diag(conditional_var) @ conditional_mean_grad efficiently
        lambda_2 =  -((-0.5 * conditional_mean_grad.T  @ (conditional_var * conditional_mean_grad)) - ( kl_partial_s))
        return lambda_2

    conditional_var = likelihood_conditional_var(model.data.X)

    lambda_2_arr = batch_or_loop(
        compute_lambda,
        [np.array(J), np.array(conditional_var), KL_grad_arr],
        [0, None, 0],
        dim = Q,
        out_dim=1,
        batch_type = get_batch_type(approx_posterior.approx_posteriors)
    )

    return lambda_2_arr
 
def gauss_newton_natural_gradient_for_mean_field_approx_posterior(model, beta, prediction_samples):
    if prediction_samples is None:
        raise RuntimeError()

    laplace_samples = 1

    m_name_list, S_chol_name_list = _get_mf_params_names(model)
    f_vars_to_diff = vc_keep_vars(model.vars(), m_name_list)
    S_vars_to_diff = vc_keep_vars(model.vars(), S_chol_name_list)
    Q = len(m_name_list)

    # keep orignal values 
    f_values = np.array(f_vars_to_diff.tensors())
    S_chol_values = np.array(S_vars_to_diff.tensors())


    M = f_values[0].shape[0]

    f_val = f_values[..., None]
    S_chol_val = vectorized_lower_triangular(S_chol_values, M)
    var_val = vectorized_cholesky_to_psd(S_chol_val)

    var_val = var_val[None, ...]
    f_val = f_val[None, ..., 0, 0]
    
    #sample 
    def wrapped_fn(f):
        chex.assert_equal(f.shape, (1, Q, M, 1))
        f_vars_to_diff.assign(f[0])

        return laplace_gauss_newton_natural_gradient_for_meanfield_approx_posterior(model, beta, laplace_samples)


    lam = mv_mean_field_block_monte_carlo(wrapped_fn, f_val, var_val, generator = model.inference.generator, num_samples = prediction_samples)


    # assign back original values
    f_vars_to_diff.assign(f_values)

    return lam

def get_mean_field_gaussian_hessian_approximation(model, beta, prediction_samples, enforce_psd_type):
    """ Hessian Approximations for Mean-Field Approximate Posteriors """
    approx_posteriors = model.approximate_posterior.approx_posteriors
    num_q = len(approx_posteriors)

    if enforce_psd_type == 'laplace_gauss_newton':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_meanfield_approx_posterior(model, beta, prediction_samples)

    elif enforce_psd_type == 'gauss_newton':
        approx_hessian =  gauss_newton_natural_gradient_for_mean_field_approx_posterior(model, beta, prediction_samples)

    else:
        approx_hessian = [None for q in range(num_q)]

    return approx_hessian

def get_full_gaussian_hessian_approximation(model, beta, prediction_samples, enforce_psd_type):
    """ Hessian Approximations for Full-Gaussian Approximate Posteriors """

    if enforce_psd_type == 'laplace_gauss_newton':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples)

    elif enforce_psd_type == 'gauss_newton':
        approx_hessian =  gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples)
    else:
        approx_hessian = None

    return approx_hessian
