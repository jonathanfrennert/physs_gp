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
from .ng_hessian_approximations import get_full_gaussian_hessian_approximation, get_mean_field_gaussian_hessian_approximation

import chex
from batchjax import batch_or_loop, BatchType
import jax
import jax.numpy as np
from jax import grad, jit, jacfwd, jacrev
from jax import vjp, jvp
from functools import partial

import objax

from typing import List

def natural_gradient_for_gaussian_approx_posterior(model, beta, approx_posterior, m_grad, s_grad):
    """
        Implments Natural gradients for q(u) with a general likelihood and Gaussian approximate posterior. 
            For further details see: 
                `Gaussian Processes for Big Data' - Hensman et al
                `Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models' - Salimbeni et al
        The approximate posterior is a Gaussian:
                q(u) = N(u | m, S), 
            where
                θ = (m, S)
            and is parameterised by
                
                ξ = (m, L) where S = LL^T
            with natural parameters:
                λ = (S⁻¹ m, - ½S⁻¹)
            and expectation parameters:
                
                μ = (m, mm^T + S⁻¹)
    """
    m = approx_posterior._m.value
    S_chol_flattened = approx_posterior._S_chol.value

    M = m.shape[0]

    S_chol = lower_triangle(S_chol_flattened, M)

    #Calculate natural and expectation parameters:
    theta_1, theta_2 = xi_to_theta(m, S_chol)
    lambda_1_init, lambda_2_init = theta_to_lambda(theta_1, theta_2)
    mu1, mu2 = xi_to_expectation(m, S_chol)

    partial_m = m_grad
    partial_s_chol_flattened = s_grad
    partial_s_chol = lower_triangle(partial_s_chol_flattened, M)

    #calculate ∂ξ/∂ 
    x, u = vjp(expectation_to_xi, mu1, mu2)

    #calculate ∂L/∂λ = ∂L/∂ξ ∂ξ/∂
    u = u((partial_m, partial_s_chol))
    lambda_1, lambda_2 = u[0], u[1]

    # Compute vector jacobian product
    #  (∂ξ/∂λ) (∂L/∂μ^T)

    #symmetrize gradient
    # This is the same problem as in gpytorch - see
    #   - https://github.com/pytorch/pytorch/issues/18825
    #   - and for the same solution `_cholesky_backward` here https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/variational/natural_variational_distribution.py 
    # TODO: double check that this is actually what is going on
    lambda_2 = lambda_2/2 
    lambda_2 = lambda_2 + lambda_2.T

    return lambda_1, lambda_2

def _natural_gradient(model, beta: float) -> np.ndarray:
    approx_posteriors = model.approximate_posterior.approx_posteriors
    num_q = len(approx_posteriors)

    param_dict = get_parameters(model, replace_name=False, return_id=True)

    m_name_list = []
    S_chol_list = []
    for q in approx_posteriors:

        m_name = get_var_name_with_id(model, id(q._m.raw_var), param_dict)
        S_chol_name = get_var_name_with_id(model, id(q._S_chol.raw_var), param_dict)

        m_name_list.append(m_name)
        S_chol_list.append(S_chol_name)

    # Precompute all gradients
    # Then vmap through them

    #calculate ∂L/∂ξ 
    vc = model.vars()

    # To use objax to compute gradients we have to pass a VarCollection
    # This requires knowning the (objax) id string
    approx_posterior_vars = [*m_name_list, *S_chol_list]

    # Extract only the approximate posterior variables to compute grads with
    vars_to_diff = vc_keep_vars(vc, approx_posterior_vars)

    # Construct the objax grad fucntions
    grad_fn = objax.GradValues(model.get_objective, vars_to_diff)
    gradients, _ = grad_fn()

    # Collect all m_grads and S_grads
    m_grads = []
    S_grads = []
    for q in range(0, len(gradients), 2):
        m_grads.append(gradients[q])
        S_grads.append(gradients[q+1])

    xi1_arr, xi2_arr = batch_or_loop(
        lambda m, b, q, m_grad, s_grad: natural_gradient_for_gaussian_approx_posterior(m, b, q, m_grad, s_grad),
        [model, beta, approx_posteriors, np.array(m_grads), np.array(S_grads)],
        [None, None, 0, 0, 0],
        dim = num_q,
        out_dim=2,
        batch_type = get_batch_type(approx_posteriors)
    )


    return xi1_arr, xi2_arr


def _natural_gradient_update_for_gaussian_approx_posterior(model, beta, approx_posterior, m_grad, s_grad, enforce_psd):
    """
        Implments Natural gradients for q(u) with a general likelihood and Gaussian approximate posterior. 
            For further details see: 
                `Gaussian Processes for Big Data' - Hensman et al
                `Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models' - Salimbeni et al
        The approximate posterior is a Gaussian:
                q(u) = N(u | m, S), 
            where
                θ = (m, S)
            and is parameterised by
                
                ξ = (m, L) where S = LL^T
            with natural parameters:
                λ = (S⁻¹ m, - ½S⁻¹)
            and expectation parameters:
                
                μ = (m, mm^T + S⁻¹)

        The natural gradient step is given by, and taking the chain rule leads to:
∂λ
∂μ
            
            ξᵣ₊₁ = ξᵣ + β (∂L/∂μ) (∂ξ/∂λ)^T

    """
    m = approx_posterior._m.value
    S_chol_flattened = approx_posterior._S_chol.value

    M = m.shape[0]

    S_chol = lower_triangle(S_chol_flattened, M)

    #Calculate natural and expectation parameters:
    lambda_1_init, lambda_2_init = xi_to_lambda(m, S_chol)
    mu1, mu2 = xi_to_expectation(m, S_chol)

    # Fix ∂L/∂ξ shapes
    partial_m = m_grad
    partial_s_chol_flattened = s_grad
    partial_s_chol = lower_triangle(partial_s_chol_flattened, M)

    # Calculate ∂L/∂μ = ∂L/∂ξ ∂ξ/∂μ

    # calculate ∂ξ/μ 
    x, u = vjp(expectation_to_xi, mu1, mu2)
    # calculate ∂L/∂ξ ∂ξ/∂μ
    res_u = u((partial_m, partial_s_chol))
    partial_mu_1, partial_mu_2 = res_u[0], res_u[1]

    # Compute (∂L/∂μ) (∂ξ/∂λ)^T
    if True:
        y, u = vjp(lambda_to_xi, lambda_1_init, lambda_2_init)
        res = u((partial_mu_1, partial_mu_2))

    if False:
        #y, res = jvp(lambda_to_xi, (lambda_1_init, lambda_2_init), (partial_mu_1, partial_mu_2))

        u_1 = np.ones_like(mu1)
        u_2 = np.ones_like(mu2)

        y1, _pb = vjp(lambda_to_xi, lambda_1_init, lambda_2_init)
        pb = lambda u1, u2: _pb((u1, u2)) 
        y2, res = jvp(pb, (u_1, u_2), (partial_mu_1, partial_mu_2))
        #res = pb((partial_mu_1, partial_mu_2))
        #pb((u_1, u_2))
        #f = lambda x: lambda_to_xi(x, lambda_1_init)

        #J  = jacfwd(lambda_to_xi, argnums=(0, 1))(lambda_1_init, lambda_2_init)

        #uu = lambda u1, u2 : jvp(lambda_to_xi, (lambda_1_init, lambda_2_init), (u1, u2))[1]
        #y1, pb = vjp(uu, u_1, u_2)
        #res_1 = pb((partial_mu_1, partial_mu_2))

    grad_1 = res[0]
    grad_2 = res[1]

    m_new = m - beta * grad_1
    S_chol_new = S_chol - beta * grad_2

    S_chol_new_vec = to_lower_triangular_vec(S_chol_new)

    return [m_new, S_chol_new_vec]

def natural_gradient_update_for_gaussian_approx_posterior(model, beta, approx_posterior, m_grad, s_grad, enforce_psd, approx_hessian = None):
    """
        Implments Natural gradients for q(u) with a general likelihood and Gaussian approximate posterior. 
            For further details see: 
                `Gaussian Processes for Big Data' - Hensman et al
                `Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models' - Salimbeni et al
        The approximate posterior is a Gaussian:
                q(u) = N(u | m, S), 
            where
                θ = (m, S)
            and is parameterised by
                
                ξ = (m, L) where S = LL^T
            with natural parameters:
                λ = (S⁻¹ m, - ½S⁻¹)
            and expectation parameters:
                
                μ = (m, mm^T + S⁻¹)
        The natural gradient step is given by, and taking the chain rule leads to:
            
            λᵣ₊₁ = λᵣ + β ∂L/∂μ
                 = λᵣ + β (∂L/∂ξ) (∂ξ/∂μ)
        This is a vector jacobian product because (∂L/∂ξ) is a vector.
        To update m, S we simply update λᵣ₊₁ and repameterise to get mᵣ₊₁, Sᵣ₊₁:
            mᵣ₊₁ = (- 2 λᵣ₊₁(2))⁻¹ λᵣ₊₁(1)
            Sᵣ₊₁ = (- 2 λᵣ₊₁(2))⁻¹
    """
    m = approx_posterior._m.value
    S_chol_flattened = approx_posterior._S_chol.value

    M = m.shape[0]

    S_chol = lower_triangle(S_chol_flattened, M)

    #Calculate natural and expectation parameters:
    theta_1, theta_2 = xi_to_theta(m, S_chol)
    lambda_1_init, lambda_2_init = theta_to_lambda(theta_1, theta_2)

    mu1, mu2 = xi_to_expectation(m, S_chol)

    partial_m = m_grad
    partial_s_chol_flattened = s_grad
    partial_s_chol = lower_triangle(partial_s_chol_flattened, M)

    #calculate ∂ξ/μ 
    x, u = vjp(expectation_to_xi, mu1, mu2)

    #calculate ∂L/μ = ∂L/∂ξ ∂ξ/μ
    u = u((partial_m, partial_s_chol))
    lambda_1, lambda_2 = u[0], u[1]

    #symmetrize gradient
    # This is the same problem as in gpytorch - see
    #   - https://github.com/pytorch/pytorch/issues/18825
    #   - and for the same solution `_cholesky_backward` here https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/variational/natural_variational_distribution.py 
    # TODO: double check that this is actually what is going on
    lambda_2 = lambda_2/2 
    lambda_2 = lambda_2 + lambda_2.T

    #gradient update
    #∂L/μ has been calculated with the negative ELBO however the natural gradients are defined on the orginal ELBO
    # hence we use the negative lambda's here
    lambda_1 = lambda_1_init + beta*(-lambda_1)

    if enforce_psd == None:
        lambda_2 = lambda_2_init + beta*(-lambda_2)
    elif enforce_psd == 'retraction':
        lambda_2 = psd_retraction_map(-2*lambda_2_init, 2*beta*lambda_2)/(-2)

    elif enforce_psd == 'riemannian':

        lambda_2_new = lambda_2_init + beta*(-lambda_2)
        lambda_to_theta(lambda_1, lambda_2)

        # update precision
        precision = -2 * lambda_2
        G = precision - grad_2

        S_chol = cholesky(add_jitter(-2*lambda_2, settings.ng_jitter))/(-2)
        A1 = cholesky_solve(S_chol, G)
        A = G @ A1
        lambda_2_new = (1-beta)*lambda_2  + beta* grad_2 + (beta **2)/2 * A

    elif enforce_psd in ['laplace_gauss_newton', 'gauss_newton']:
        # use gauss newton approximation of lambda_2
        lambda_2 = lambda_2_init + beta*(-approx_hessian)
    else:
        raise RuntimeError(f'enforce_psd of type {enforce_psd} unknown')


    #convert from natural parameters to the raw parameters
    theta_1, theta_2 = lambda_to_theta(lambda_1, lambda_2)

    xi1, xi2 = theta_to_xi(theta_1, theta_2)
    xi2 = xi2[np.tril_indices(M, 0)]

    return [xi1, xi2]





@dispatch('VGP', 'MeanFieldApproximatePosterior')
@dispatch('MultiObjectiveModel', 'MeanFieldApproximatePosterior')
def natural_gradients(model, beta: float, enforce_psd_type, prediction_samples=None) -> np.ndarray:
    approx_posteriors = model.approximate_posterior.approx_posteriors
    num_q = len(approx_posteriors)

    m_name_list, S_chol_list = _get_mf_params_names(model)

    # Precompute all gradients
    # Then vmap through them

    #calculate ∂L/∂ξ 
    vc = model.vars()

    # To use objax to compute gradients we have to pass a VarCollection
    # This requires knowning the (objax) id string
    approx_posterior_vars = [*m_name_list, *S_chol_list]

    # Extract only the approximate posterior variables to compute grads with
    vars_to_diff = vc_keep_vars(vc, approx_posterior_vars)

    # Construct the objax grad fucntions
    grad_fn = objax.GradValues(model.get_objective, vars_to_diff)
    gradients, _ = grad_fn()

    # Collect all m_grads and S_grads
    m_grads = []
    S_grads = []
    for q in range(0, len(gradients), 2):
        m_grads.append(gradients[q])
        S_grads.append(gradients[q+1])

    if type(beta) is list:
        beta_axis = 0
        beta = np.array(beta)
    else:
        beta_axis = None

    approx_hessian = get_mean_field_gaussian_hessian_approximation(model, beta, prediction_samples, enforce_psd_type)

    xi1_arr, xi2_arr = batch_or_loop(
        lambda m, b, q, m_grad, s_grad, enforce_psd, approx_hessian: natural_gradient_update_for_gaussian_approx_posterior(m, b, q, m_grad, s_grad, enforce_psd, approx_hessian),
        [model, beta, approx_posteriors, np.array(m_grads), np.array(S_grads), enforce_psd_type, approx_hessian],
        [None, beta_axis, 0, 0, 0, None, 0],
        dim = num_q,
        out_dim=2,
        batch_type = BatchType.LOOP
    )

    return xi1_arr, xi2_arr

@dispatch('VGP', 'FullGaussianApproximatePosterior')
def natural_gradients(model, beta: float, enforce_psd_type, prediction_samples=None) -> np.ndarray:
    q = model.approximate_posterior

    param_dict = get_parameters(model, replace_name=False, return_id=True)

    # Collect q parameters
    m_name = get_var_name_with_id(model, id(q._m.raw_var), param_dict)
    S_chol_name = get_var_name_with_id(model, id(q._S_chol.raw_var), param_dict)

    approx_posterior_vars = [m_name, S_chol_name]

    vc = model.vars()

    # Extract only the approximate posterior variables to compute grads with
    vars_to_diff = vc_keep_vars(vc, approx_posterior_vars)

    # Construct the objax grad fucntions
    grad_fn = objax.GradValues(model.get_objective, vars_to_diff)
    gradients, val = grad_fn()
    m_grad = gradients[0]
    S_grad = gradients[1]

    approx_hessian = get_full_gaussian_hessian_approximation(model, beta, prediction_samples, enforce_psd_type)

    xi1, xi2 = natural_gradient_update_for_gaussian_approx_posterior(model, beta, q, m_grad, S_grad, enforce_psd_type, approx_hessian)

    return xi1, xi2






