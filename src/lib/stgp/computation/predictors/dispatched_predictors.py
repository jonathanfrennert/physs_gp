"""
Dispatched Prediction Functions.

In general each function should return two items:

    - mu: rank 3: N x P x A
    - var: rank 4: N x P x A x A or N x 1 x PA x PA etc

where

    - N: number of data points
    - P: number of outputs
    - A: block size
"""
# Import Types
from ...data import Data
from ...kernels import Kernel, RBF
from ...likelihood import Gaussian, GaussianParameterised, ProductLikelihood, DiagonalGaussian, Likelihood, BlockDiagonalGaussian, PrecisionBlockDiagonalGaussian
from ...approximate_posteriors import GaussianApproximatePosterior, MeanFieldApproximatePosterior, ApproximatePosterior
from ...dispatch import dispatch, evoke
from ..gaussian import log_gaussian
from ...transforms import Independent, Transform, LinearTransform, NonLinearTransform, Aggregate
from ..permutations import data_order_to_output_order, permute_mat_ld_to_dl, permute_mat_dl_to_ld, permute_mat_tps_to_tsp, permute_vec_tps_to_tsp, permute_vec_dl_to_ld, permute_mat_tsp_to_tps

from ...utils import utils
from ...utils.utils import can_batch, get_batch_type
from ...utils.batch_utils import batch_over_module_types
from ...utils.nan_utils import mask_to_identity, get_mask, mask_vector

from ...dispatch import _ensure_str

from ..matrix_ops import cholesky, log_chol_matrix_det, add_jitter, cholesky_solve, vec_columns, get_block_diagonal, stack_rows, to_block_diag, vec_rows
from ..model_ops import get_diagonal_gaussian_likelihood_variances

from .base_predictors import gaussian_prediction, gaussian_predictive_covar, gaussian_predictive_mean, gaussian_prediction_diagonal, gaussian_prediction_blocks, gaussian_prediction_diagonal_with_additive_noise_precision, gaussian_prediction_with_additive_noise_precision

import jax
from jax import jit
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList
from batchjax import batch_or_loop, BatchType



# =========================== Likelihood specific GPR prediction equations ===========================

@dispatch('BatchGP', PrecisionBlockDiagonalGaussian)
@dispatch('BatchGP', BlockDiagonalGaussian)
def predict_diagonal(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs, block_size):
    NS = XS.shape[0]
    N = X.shape[0]
    chex.assert_equal(K_xx.shape, (N, N))
    chex.assert_equal(K_xs_x.shape, (NS, N))
    chex.assert_equal(K_xs.shape, (NS, ))
    chex.assert_rank(Y, 2)

    chex.assert_equal(Y.shape[1], likelihood.block_size)

    Y_vec = vec_columns(Y)

    if _ensure_str(likelihood) == 'BlockDiagonalGaussian':
        # Convert Gaussian likelihood noise to full matrix
        lik_var = likelihood.full_variance
        mu, var = gaussian_prediction_diagonal(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    elif _ensure_str(likelihood) == 'PrecisionBlockDiagonalGaussian':
        # Convert Gaussian likelihood noise to full matrix
        lik_inv_var = likelihood.full_precision
        mu, var = gaussian_prediction_diagonal_with_additive_noise_precision(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_inv_var)
    else:
        raise RuntimeError()

    # this function only supports one output, but for compatability add the extra dimensions
    mu = mu[..., None]
    var = var[..., None, None]

    chex.assert_rank([mu, var], [3, 4])
    return mu, var


@dispatch('BatchGP', PrecisionBlockDiagonalGaussian)
@dispatch('BatchGP', BlockDiagonalGaussian)
def predict_full(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs, block_size):
    NS = XS.shape[0]
    N = X.shape[0]
    chex.assert_equal(K_xx.shape, (N, N))
    chex.assert_equal(K_xs_x.shape, (NS, N))
    chex.assert_equal(K_xs.shape, (NS, NS))
    chex.assert_rank(Y, 2)

    chex.assert_equal(Y.shape[1], likelihood.block_size)

    Y_vec = vec_columns(Y)

    if _ensure_str(likelihood) == 'BlockDiagonalGaussian':
        lik_var = likelihood.full_variance
        mu, var = gaussian_prediction(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)
    elif _ensure_str(likelihood) == 'PrecisionBlockDiagonalGaussian':
        lik_var_inv = likelihood.full_precision
        mu, var = gaussian_prediction_with_additive_noise_precision(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var_inv)
    else:
        raise RuntimeError()

    # this function only supports one output, but for compatability add the extra dimensions
    if block_size == NS:
        # block size is NS so the shape of mu should be 1 x 1 x Ns
        mu = (mu.T)[None, ...]
        var = var[None, None, ...]
    else:
        raise NotImplementedError()

    chex.assert_rank([mu, var], [3, 4])

    return mu, var

@dispatch('BatchGP', DiagonalGaussian)
@dispatch('BatchGP', Gaussian)
def predict_full(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs, block_size):
    NS = XS.shape[0]
    N = X.shape[0]
    chex.assert_equal(K_xx.shape, (N, N))
    chex.assert_equal(K_xs_x.shape, (NS, N))
    chex.assert_equal(K_xs.shape, (NS, NS))
    chex.assert_rank(Y, 2)

    if _ensure_str(likelihood) == 'DiagonalGaussian':
        # Convert Gaussian likelihood noise to diagonal matrix
        lik_var = likelihood.variance
    elif _ensure_str(likelihood) == 'Gaussian':
        # Convert Gaussian likelihood noise to diagonal matrix
        lik_var = np.eye(N) * likelihood.variance
    else:
        raise RuntimeError()

    mu, var = gaussian_prediction(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    # this function only supports one output, but for compatability add the extra dimensions
    mu = (mu.T)[None, ...]
    #mu = mu[None, ...]
    var = var[None, None, ...]

    chex.assert_rank([mu, var], [3, 4])
    return mu, var


@dispatch('BatchGP', DiagonalGaussian)
@dispatch('BatchGP', Gaussian)
def predict_diagonal(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs, block_size):
    NS = XS.shape[0]
    N = X.shape[0]
    chex.assert_equal(K_xx.shape, (N, N))
    chex.assert_equal(K_xs_x.shape, (NS, N))
    chex.assert_equal(K_xs.shape, (NS, ))
    chex.assert_rank(Y, 2)

    if _ensure_str(likelihood) == 'DiagonalGaussian':
        # Convert Gaussian likelihood noise to diagonal matrix
        lik_var = likelihood.variance
    elif _ensure_str(likelihood) == 'Gaussian':
        # Convert Gaussian likelihood noise to diagonal matrix
        lik_var = np.eye(N) * likelihood.variance
    else:
        raise RuntimeError()

    mu, var = gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    # this function only supports one output, but for compatability add the extra dimensions
    mu = mu[..., None]
    var = var[..., None, None]

    chex.assert_rank([mu, var], [3, 4])
    return mu, var

@dispatch('BatchGP', BlockDiagonalGaussian)
@dispatch('BatchGP', DiagonalGaussian)
@dispatch('BatchGP', Gaussian)
def predict_covar(XS_1, XS_2, X, Y, likelihood, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs):

    if _ensure_str(likelihood) == 'Gaussian':
        lik_var = np.eye(K_xx.shape[0]) * likelihood.variance
    elif _ensure_str(likelihood) in ['BlockDiagonalGaussian', 'DiagonalGaussian']:
        lik_var = likelihood.full_variance
    else:
        raise RuntimeError()

    return  gaussian_predictive_covar(Y, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs, lik_var)


@dispatch(Data, 'BatchGP', ProductLikelihood, Independent)
def predict_covar(XS_1, XS_2, data, gp, likelihood, prior):
    X = data.X
    Y = data.Y

    num_latents = prior.output_dim
    num_outputs = prior.output_dim

    # precompute batched kernels
    
    K_xs = prior.covar_blocks(XS_1, XS_2)
    K_xx = prior.covar_blocks(X, X)
    K_xs_x = prior.covar_blocks(XS_1, X)
    K_x_xs = prior.covar_blocks(X, XS_2)
    mean_x = prior.mean_blocks(X)
    mean_xs_1 = prior.mean_blocks(XS_1)
    mean_xs_2 = prior.mean_blocks(XS_2)

    likelihood_arr = likelihood.likelihood_arr

    # Ensure Y is rank 2 after batching
    Y = Y[..., None]

    var_arr =  batch_over_module_types(
        'predict_covar',
        [gp],
        likelihood_arr,
        [XS_1, XS_2, X, Y, likelihood_arr, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs_1],
        [None, None, None, 1, 0, 0, 0, 0, 0, 0, 0],
        num_latents,
        1
    )

    var_arr = np.array(var_arr)
    chex.assert_shape(var_arr, [num_outputs, XS_1.shape[0], XS_2.shape[0]])

    return var_arr


@dispatch(Data, 'BatchGP', ProductLikelihood, Independent)
def predict_blocks(XS, data, gp, likelihood, prior, block_size: int):
    """ 
    An Independent prior with a product likelihood are treated as separate models and batched over.
    """
    X = data.X
    Y = data.Y

    NS = XS.shape[0]
    N, P = Y.shape[0], Y.shape[1]
    num_outputs = prior.output_dim

    if block_size == 1:
        K_xs = prior.var_blocks(XS)
        # returns rank 3 but we need rank 2
        K_xs = K_xs[..., 0]
        evoke_name = 'predict_diagonal'
    elif block_size == NS:
        K_xs = prior.covar_blocks(XS, XS)
        evoke_name = 'predict_full'
    else:
        K_xs = prior.covar_blocks(XS, XS)
        evoke_name = 'predict_blocks'

    # Precompute batched kernels
    K_xx = prior.covar_blocks(X, X)
    K_xs_x = prior.covar_blocks(XS, X)
    mean_x = prior.mean_blocks(X)
    mean_xs = prior.mean_blocks(XS)

    likelihood_arr = likelihood.likelihood_arr

    # Ensure Y is rank 2 after batching
    if len(Y.shape) == 2:
        Y = Y[..., None]

    # Batch across all latents
    marginal_mu, marginal_var =  batch_over_module_types(
        evoke_name,
        [gp],
        likelihood_arr,
        [XS, X, Y, likelihood_arr, K_xs, K_xs_x, K_xx, mean_x, mean_xs, block_size],
        [None, None, 1, 0, 0, 0, 0, 0, 0, None],
        num_outputs,
        2
    )
    marginal_mu = np.array(marginal_mu)
    marginal_var = np.array(marginal_var)

    V_P, V_NS, _, V_B, _ = marginal_var.shape

    # fix shapes
    # each component will return rank (3, 4). But each component is only one ouput so we can remove that axis
    #   and reshape into the proper shape
    marginal_mu = marginal_mu[:, :, 0, ...]
    marginal_mu = np.transpose(marginal_mu, [1, 0, 2])
    chex.assert_shape(marginal_mu, [V_NS, num_outputs,  block_size])

    if V_NS == 1:
        # full prediction
        marginal_var = marginal_var[:, :, 0, ...]
        marginal_var = np.transpose(marginal_var, [1, 0, 2, 3])
        chex.assert_shape(marginal_var, [1, num_outputs, NS, NS])
    else:
        marginal_var = marginal_var[..., 0]
        marginal_var = np.transpose(marginal_var, [1, 0, 2, 3])
        # Mean field so we do not capture the correlations between Q
        chex.assert_shape(marginal_var, [NS, num_outputs, block_size, block_size])

    return marginal_mu, marginal_var


@dispatch(Data, 'BatchGP', ProductLikelihood, LinearTransform)
def predict_blocks(XS, data, gp, likelihood, prior, block_size: int):
    """ 
    A linear model is treated as a full joint model. To compute we stack Y and treat like a standard
       Gaussian
    """

    X = data.X
    Y = data.Y

    NS = XS.shape[0]
    N, P = Y.shape
    num_outputs = prior.output_dim

    # Computed stacked K and Y
    likelihood_arr = likelihood.likelihood_arr

    K_xx = prior.covar(X, X)
    K_xs_x = prior.covar(XS, X)
    mean_x = prior.mean(X)
    mean_xs = prior.mean(XS)

    Y_vec = vec_columns(Y)

    # TODO: generalise to different likelihoods
    lik_var = get_diagonal_gaussian_likelihood_variances(Y, likelihood_arr)

    if block_size == 1:
        K_xs = prior.var(XS)[..., 0]
        mu, var = gaussian_prediction_diagonal(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

        # Convert from stacked and fix shapes
        mu = np.reshape(mu, [P, NS, 1])
        var = np.reshape(var, [P, NS, 1])
        mu = np.transpose(mu, [1, 0, 2])
        var = np.transpose(var, [1, 0, 2])[..., None]

    elif block_size == NS:
        # TODO: need to decide on a convention here, difference between returning 
        #   full
        #   blocks of size NS
        #   blocks of size P
        # Not all of this information can be stored in block_size
        #  Need a prediction type? or maybe use a string?
        K_xs = prior.covar(XS, XS)
        mu, var =  gaussian_prediction(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)
        mu = np.reshape(mu, [P, NS, 1])
        var = var[None, None, ...]
    else:
        raise NotImplementedError()


    chex.assert_rank([mu, var], [3, 4])
    return mu, var

@dispatch(Data, 'BatchGP', BlockDiagonalGaussian, Independent)
@dispatch(Data, 'BatchGP', BlockDiagonalGaussian, LinearTransform)
def predict_blocks(XS, data, gp, likelihood, prior, block_size: int):
    """
    Compute the posterior with the same block structure as the likelihood

    p(vec(F)) ~ GP() in latent-data format 
    X in (time-space) 
    Y in data-latent
    vec(Y) in latent-data

    """
    X = data.X
    Y = data.Y

    NS = XS.shape[0]
    N = X.shape[0]
    P = Y.shape[1]
    Q = prior.output_dim
    num_latents = likelihood.num_latents

    #chex.assert_equal([NS], [N])
    #chex.assert_equal([block_size], [likelihood.block_size])

    # hack for now
    block_size = likelihood.block_size
    chex.assert_equal(block_size, P)

    # compute prior covariances in latent-data format
    K_xs = prior.covar(XS, XS) 
    K_xx = prior.covar(X, X) 
    K_xs_x = prior.covar(XS, X) 

    # Get liklihood in data-latent format
    likelihood_var = likelihood.full_variance

    # Permute so that the ordering between likelihood_var and Y is the same

    # convert likelihodo to latent-data format
    #likelihood is time - latent - space format
    # convert to data - latent format

    lik_var_bd_tsp = permute_mat_tps_to_tsp(likelihood.variance, likelihood.num_latents)
    lik_var_tsp = to_block_diag(lik_var_bd_tsp)

    # convert to latent-data
    lik_var = permute_mat_dl_to_ld(lik_var_tsp, likelihood.num_latents, N)

    mean_x = prior.mean(X)
    mean_xs = prior.mean(XS)

    # Y is in time-(space)-latent, convert to latent-data
    Y_vec = permute_vec_dl_to_ld(np.reshape(Y, [-1, 1]), likelihood.num_latents, N) 

    # TODO: this is v. inefficient
    # Compute full matrix in latent-data format
    mu, var = gaussian_prediction(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    # convert K from latent-data to data-latent format
    K = permute_mat_ld_to_dl(var, likelihood.num_latents, NS) 
    var = get_block_diagonal(K, likelihood.block_size)

    # we need ot return in time-latent-space format (to mimic the kalman filter)
    var = permute_mat_tsp_to_tps(var, likelihood.num_latents)

    mu = np.reshape(mu, [likelihood.num_latents, likelihood.num_blocks, -1])
    mu = np.reshape(np.transpose(mu, [1, 0, 2]), [likelihood.num_blocks, likelihood.block_size])

    mu = mu[..., None]

    var = var[:, None, ...]

    chex.assert_rank([mu, var], [3, 4])


    return mu, var


# =========================== Model Specific prediction equations ===========================
@dispatch(Data, 'BatchGP', Likelihood, LinearTransform)
@dispatch(Data, 'BatchGP', Likelihood, Independent)
def predict(XS, data, gp, likelihood, prior, diagonal: bool):
    """ Only supports linear models.  """

    if diagonal:
        block_size = 1
    else:
        block_size = XS.shape[0]
        
    return evoke('predict_blocks', data, gp, likelihood, prior)(
        XS, data, gp, likelihood, prior, block_size
    )
    

# =========================== Entry Point For Variational Predictions ===========================

@dispatch(Likelihood, Transform, ApproximatePosterior, True)
@dispatch(Likelihood, Transform, ApproximatePosterior, False)
def predict(XS, data, likelihood, prior, approximate_posterior, inference, whiten, diagonal, **kwargs):
    return  evoke('marginal_prediction', approximate_posterior, likelihood, prior, whiten=whiten)(
        XS, data, approximate_posterior, likelihood, prior, inference, diagonal, whiten, **kwargs
    )
