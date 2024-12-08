import jax
import jax.numpy as np
from jax import  grad, jit, jacfwd, vjp
import chex
import objax
from batchjax import batch_or_loop, BatchType
from functools import partial

from ... import settings
from ...utils.nan_utils import get_same_shape_mask 
from ..matrix_ops import cholesky, cholesky_solve, triangular_solve, vec_add_jitter, add_jitter, lower_triangle, vectorized_lower_triangular_cholesky, vectorized_lower_triangular, lower_triangular_cholesky, lower_triangle
from ...utils.utils import vc_keep_vars, get_parameters, get_var_name_with_id, get_batch_type
from ..elbos.elbos import compute_expected_log_liklihood, compute_expected_log_liklihood_with_variational_params
from ...dispatch import dispatch, evoke
from ..parameter_transforms import psd_retraction_map
from ..integrals.samples import _process_samples
from ..integrals.approximators import mv_block_monte_carlo
from ..permutations import data_order_to_output_order, permute_mat, permute_vec

from .cvi_hessian_approximations import get_full_gaussian_hessian_approximation
from .parameterisations import get_parameterisation_class 

from ...dispatch import _ensure_str

# Types imports
from ...approximate_posteriors import ConjugateApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullConjugateGaussian, FullGaussianApproximatePosterior, DataLatentBlockDiagonalApproximatePosterior, ApproximatePosterior, DiagonalGaussianApproximatePosterior, MeanFieldConjugateGaussian
from ...sparsity import NoSparsity, FreeSparsity, Sparsity, SpatialSparsity

from .exponential_family_transforms import xi_to_theta, theta_to_lambda, xi_to_expectation, expectation_to_xi, lambda_to_theta, theta_to_xi, theta_to_lambda_diagonal, lambda_to_theta_diagonal, reparametise_cholesky_grad, lambda_to_theta_precision, theta_precision_to_lambda

from ...transforms import MultiOutput

@jit
def reparametise_vec_grad(m, m_grad, prior):
    #Calculate ∂L/μ = ∂L/∂ξ ∂ξ/μ 
    x, u = vjp(
        lambda v: prior.unpermute_vec(v), 
        m
    )
    m_grad = u(m_grad)[0]

    return m_grad


def _get_fp_params(q_mu_z, model, parameterisation):
    q = model.approximate_posterior

    # raw_Y_arr = [time -latent -space]
    raw_Y_arr = q.surrogate.data._Y.value


    #raw_Y_arr is in time-latent-space format, this reshape will preserve that
    Y_tilde_arr = np.reshape(raw_Y_arr, q_mu_z.shape)

    theta_1 = Y_tilde_arr
    theta_2 = q.surrogate.likelihood.variance

    print('theta_1: ', np.sum(theta_1), 'theta_2: ', np.sum(theta_2))

    if _ensure_str(parameterisation) == 'NG_Moment':
        lambda_1_arr, lambda_2_arr = jax.vmap(theta_to_lambda)(theta_1, theta_2)
    elif _ensure_str(parameterisation) == 'NG_Precision':
        lambda_1_arr, lambda_2_arr = jax.vmap(theta_precision_to_lambda)(Y_tilde_arr, q.surrogate.likelihood.precision)
    else:
        raise RuntimeError()

    #breakpoint()

    return raw_Y_arr, lambda_1_arr, lambda_2_arr

def _get_mf_params(model, parameterisation, diagonal=True):
    """ Helper function to wrap up batching over the latent GPs to collect variational parameters"""
    q = model.approximate_posterior
    # TODO: this should only return lambdas

    # Get natural parameters
    q_list = model.approximate_posterior.approx_posteriors
    Q = len(q_list)

    def get_likelihood(lik):
        try:
            return lik.likelihood_arr[0]
        except Exception as e:
            return lik

    # Collect CVI parameters
    if _ensure_str(parameterisation) == 'NG_Moment':
        raw_Y, Y_tilde_arr, V_tilde_arr = batch_or_loop(
            lambda q: (q.surrogate.data.base._Y.value, q.surrogate.data.base.Y, get_likelihood(q.surrogate.likelihood).base.variance),
            [q_list],
            [0],
            dim=len(q_list),
            out_dim=2,
            batch_type = get_batch_type(q_list)
        )
    elif _ensure_str(parameterisation) == 'NG_Precision':
        raw_Y, Y_tilde_arr, V_tilde_arr = batch_or_loop(
            lambda q: (q.surrogate.data.base._Y.value, q.surrogate.data.base.Y, get_likelihood(q.surrogate.likelihood).base.precision),
            [q_list],
            [0],
            dim=len(q_list),
            out_dim=2,
            batch_type = get_batch_type(q_list)
        )

    if diagonal:
        fn = lambda q: q.surrogate.posterior(diagonal=True)
    else:
        fn = lambda q: q.surrogate.posterior_blocks()

    # Compute q_m_z, q_S_z
    q_mu_z, q_var_z = batch_or_loop(
        fn,
        [q_list],
        [0],
        dim = Q,
        out_dim=2,
        batch_type = get_batch_type(q_list)
    )

    # each component returns rank [3, 4]. Remove the extra dimension and fix ordering.
    q_mu_z = np.transpose(q_mu_z[..., 0], [1, 0, 2])
    q_var_z = np.transpose(q_var_z[:, :, 0, ...], [1, 0, 2, 3])

    if _ensure_str(parameterisation) == 'NG_Moment':
        if True:
            # convert momement parameters to [Nt, Q, Ns x L] format (the same as q_mu_z)
            Nt, Q, NtL, _ = V_tilde_arr.shape
            _Y =  np.reshape(Y_tilde_arr, [Nt, Q, NtL, 1])
            # convert to natural parameters
            lambda_1_arr, lambda_2_arr = jax.vmap(jax.vmap(theta_to_lambda))(_Y, V_tilde_arr)
            # remove  extra dim so that lambda_1 is rank 3
            lambda_1_arr = lambda_1_arr[..., 0]
        else:
            lambda_1_arr, lambda_2_arr = jax.vmap(jax.vmap(theta_to_lambda))(Y_tilde_arr, V_tilde_arr)

    elif _ensure_str(parameterisation) == 'NG_Precision':
        lambda_1_arr, lambda_2_arr = jax.vmap(jax.vmap(theta_precision_to_lambda))(Y_tilde_arr, V_tilde_arr)
    else:
        raise RuntimeError()

    # convert Y, V to natural parameters

    return raw_Y, lambda_1_arr, lambda_2_arr, q_mu_z, q_var_z

def _get_marginals(model):
    q_m, q_S = evoke('variational_params', model.approximate_posterior, model.likelihood, model.prior.base_prior, model.inference.whiten)(
        model.data, model.approximate_posterior, model.likelihood, model.prior, model.inference.whiten
    )

    q_f_mu, q_f_var = evoke('marginal', model.approximate_posterior, model.likelihood, model.prior, whiten=model.inference.whiten)(
        model.data, q_m, q_S, model.approximate_posterior, model.likelihood, model.prior, model.inference.whiten
    )
    return q_f_mu, q_f_var 

def partial_ell(m, q_m, q_S):
    """ Helper function to compute the expected log likelihood using the variational paramters q_m, q_S"""

    # q_m, q_S should be in time-latent-space
    return compute_expected_log_liklihood_with_variational_params(
        m.data,
        q_m,
        q_S,
        m.likelihood, 
        m.prior,
        m.approximate_posterior,
        m.inference
    )

