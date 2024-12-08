"""
Dispatched Prediction Functions.

In general each function should return two items:

    - mu: rank 3: N x P x A
    - var: rank 4: N x P x A x A or N x 1 x PA x PA

where

    - N: number of data points
    - P: number of outputs
    - A: block size
"""
# Import Types
from ...data import Data
from ...kernels import Kernel, RBF
from ...likelihood import Gaussian, GaussianParameterised, ProductLikelihood, DiagonalGaussian, Likelihood, BlockDiagonalGaussian
from ...approximate_posteriors import GaussianApproximatePosterior, MeanFieldApproximatePosterior, ApproximatePosterior
from ...dispatch import dispatch, evoke
from ..gaussian import log_gaussian
from ...transforms import Independent, Transform, LinearTransform, NonLinearTransform, Aggregate
from ..permutations import data_order_to_output_order

from ...utils import utils
from ...utils.utils import can_batch, get_batch_type
from ...utils.batch_utils import batch_over_module_types
from ...utils.nan_utils import mask_to_identity, get_mask, mask_vector

from ..matrix_ops import cholesky, log_chol_matrix_det, add_jitter, cholesky_solve, vec_columns, get_block_diagonal, stack_rows
from ..model_ops import get_diagonal_gaussian_likelihood_variances

from .base_predictors import gaussian_prediction, gaussian_predictive_covar, gaussian_predictive_mean, gaussian_prediction_diagonal, gaussian_prediction_blocks

import jax
from jax import jit
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList
from batchjax import batch_or_loop, BatchType


# =========================== Likelihood specific GPR prediction equations ===========================
@dispatch('BatchGP', Gaussian)
def predict(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs):
    NS = XS.shape[0]
    N = X.shape[0]
    chex.assert_equal(K_xx.shape, (N, N))
    chex.assert_equal(K_xs_x.shape, (NS, N))
    chex.assert_equal(K_xs.shape, (NS, NS))

    lik_var = np.eye(K_xx.shape[0]) * likelihood.variance

    mu, var =  gaussian_prediction(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    breakpoint()
    chex.assert_rank([mu, var], [3, 4])
    return mu, var

@dispatch('BatchGP', BlockDiagonalGaussian)
def predict(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs):
    chex.assert_equal(K_xx.shape, (X.shape[0], X.shape[0]))
    chex.assert_equal(K_xs_x.shape, (XS.shape[0], X.shape[0]))
    chex.assert_equal(K_xs.shape, (XS.shape[0], XS.shape[0]))

    lik_var = likelihood.full_variance

    return gaussian_prediction(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

@dispatch('BatchGP', Gaussian)
def predict_diagonal(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs):
    NS = XS.shape[0]
    N = X.shape[0]
    chex.assert_equal(K_xx.shape, (N, N))
    chex.assert_equal(K_xs_x.shape, (NS, N))
    chex.assert_equal(K_xs.shape, (NS, ))

    # Convert Gaussian likelihood noise to diagonal matrix
    lik_var = np.eye(N) * likelihood.variance

    mu, var = gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    # this function only supports one output, but for compatability add the extra dimensions
    mu = mu[..., None]
    var = var[..., None, None]

    chex.assert_rank([mu, var], [3, 4])
    return mu, var

@dispatch('BatchGP', BlockDiagonalGaussian)
def predict_diagonal(XS, X, Y, likelihood, K_xs, K_xs_x, K_xx, mean_x, mean_xs):
    chex.assert_equal(K_xx.shape, (X.shape[0], X.shape[0]))
    chex.assert_equal(K_xs_x.shape, (XS.shape[0], X.shape[0]))
    chex.assert_shape(K_xs, [XS.shape[0]])

    lik_var = likelihood.full_variance

    return gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

@dispatch('BatchGP', Gaussian)
def predict_covar(XS_1, XS_2, X, Y, likelihood, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs):
    lik_var = np.eye(K_xx.shape[0]) * likelihood.variance

    return  gaussian_predictive_covar(Y, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs, lik_var)

# =========================== Model Specific prediction equations ===========================

@dispatch(Data, 'BatchGP', ProductLikelihood, Independent)
def predict(XS, data, gp, likelihood, prior, diagonal: bool):
    X = data.X
    Y = data.Y

    num_outputs = prior.output_dim

    if diagonal:
        K_xs = prior.var_blocks(XS)
        # returns rank 3 but we need rank 2
        K_xs = K_xs[..., 0]
        evoke_name = 'predict_diagonal'
    else:
        K_xs = prior.covar_blocks(XS, XS)
        evoke_name = 'predict'

    # Precompute batched kernels
    K_xx = prior.covar_blocks(X, X)
    K_xs_x = prior.covar_blocks(XS, X)

    mean_x = prior.mean_blocks(X)
    mean_xs = prior.mean_blocks(XS)

    likelihood_arr = likelihood.likelihood_arr

    # Ensure Y is rank 2 after batching
    Y = Y[..., None]

    mu_arr, var_arr =  batch_over_module_types(
        evoke_name,
        [gp],
        likelihood_arr,
        [XS, X, Y, likelihood_arr, K_xs, K_xs_x, K_xx, mean_x, mean_xs],
        [None, None, 1, 0, 0, 0, 0, 0, 0],
        num_outputs,
        2
    )

    Ns = XS.shape[0]
    P = Y.shape[1]

    mu_arr = np.transpose(mu_arr, [1, 0, 2])

    if diagonal:
        var_arr = np.transpose(var_arr, [1, 0, 2])[..., None]
    else:
        raise NotImplementedError()
        var_arr = var_arr.reshape([P, Ns, Ns])

    return mu_arr, var_arr

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

    chex.assert_shape(var_arr, [num_outputs, XS_1.shape[0], XS_2.shape[0]])

    return var_arr

@dispatch(Data, 'BatchGP', ProductLikelihood, LinearTransform)
def predict(XS, data, gp, likelihood, prior, diagonal):
    X = data.X
    Y = data.Y

    NS = XS.shape[0]
    N = X.shape[0]
    P = Y.shape[1]

    likelihood_arr = likelihood.likelihood_arr

    K_xs = prior.var(XS)[..., 0]
    K_xx = prior.covar(X, X)
    K_xs_x = prior.covar(XS, X)
    lik_var = get_diagonal_gaussian_likelihood_variances(Y, likelihood_arr)
    mean_x = prior.mean(X)
    mean_xs = prior.mean(XS)

    Y_vec = vec_columns(Y)

    if diagonal:
        mu, var = gaussian_prediction_diagonal(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)
        mu = np.reshape(mu, [P, NS, 1])
        var = np.reshape(var, [P, NS, 1])

        mu = np.transpose(mu, [1, 0, 2])
        var = np.transpose(var, [1, 0, 2])[..., None]

    else:
        mu, var = gaussian_prediction(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)
        raise NotImplementedError()


    return mu, var

@dispatch(Data, 'BatchGP', BlockDiagonalGaussian, LinearTransform)
def predict(XS, data, gp, likelihood, prior, diagonal):
    X = data.X
    Y = data.Y

    Ns = XS.shape[0]
    N = X.shape[0]
    P = Y.shape[1]

    K_xx = prior.full_covar(X, X)
    K_xs_x = prior.full_covar(XS, X)

    # Get liklihood
    likelihood_var = jax.scipy.linalg.block_diag(*likelihood.variance)

    # Permute so that the ordering between likelihood_var and Y is the same
    N = X.shape[0]

    permutation = data_order_to_output_order(P, N)

    lik_var = permutation @ likelihood_var @ permutation.T

    mean_x = prior.vec_mean(X)
    mean_xs = prior.vec_mean(XS)

    Y_vec = vec_columns(Y)

    if diagonal:
        K_xs = prior.vec_var(XS)[:, 0]

        mu, var = gaussian_prediction_diagonal(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

        mu = mu.reshape([P, Ns])
        var = var.reshape([P, Ns])
    else:
        K_xs = prior.full_covar(XS, XS)

        mu, var = gaussian_prediction(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)


    return mu, var

@dispatch(Data, 'BatchGP', BlockDiagonalGaussian, LinearTransform)
def predict_blocks(XS, data, group_size, block_size, gp, likelihood, prior, diagonal):
    X = data.X
    Y = data.Y

    Ns = XS.shape[0]
    N = X.shape[0]
    P = Y.shape[1]

    K_xs = prior.full_covar(XS, XS)
    K_xx = prior.full_covar(X, X)
    K_xs_x = prior.full_covar(XS, X)

    # Get liklihood
    likelihood_var = likelihood.full_variance

    # Permute so that the ordering between likelihood_var and Y is the same
    N = X.shape[0]
    NS = likelihood_var.shape[0]


    permutation = data_order_to_output_order(P, N)
    lik_var = permutation @ likelihood_var @ permutation.T

    mean_x = prior.vec_mean(X)
    mean_xs = prior.vec_mean(XS)

    Y_vec = vec_columns(Y)

    # TODO: this is v. inefficient
    # Compute full matrix
    mu, var = gaussian_prediction(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    NS = var.shape[0]
    N = XS.shape[0]
    i = np.hstack([np.arange(i , NS, N) for i in range(N)])
    permutaton = np.eye(NS)[i]

    K = permutaton @ var @ permutaton.T

    var = get_block_diagonal(K, likelihood.block_size)

    mu = mu.reshape([P, Ns]).T

    return mu, var

@dispatch(Data, 'BatchGP', 'BlockGaussianProductLikelihood', 'DataLatentPermutation')
def predict(XS, data, gp, likelihood, prior, diagonal):
    X = data.X
    Y = data.Y

    likelihood = likelihood.likelihood_arr[0]

    if diagonal:
        K_xs = prior.vec_var(XS)[0]
    else:
        K_xs = prior.full_covar(XS, XS)

    K_xx = prior.full_covar(X, X)
    K_xs_x = prior.full_covar(XS, X)

    # Get liklihood
    lik_var = jax.scipy.linalg.block_diag(*likelihood.variance)

    mean_x = prior.vec_mean(X)
    mean_xs = prior.vec_mean(XS)

    Y_vec = vec_columns(Y)
    
    if diagonal:
        mu, var = gaussian_prediction_diagonal(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)
    else:
        mu, var = gaussian_prediction(Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    return mu, var

@dispatch(Data, 'BatchGP', 'BlockDiagonalGaussian', 'DataLatentPermutation')
def predict_blocks(XS, data, group_size, block_size, gp, likelihood, prior, diagonal):
    X = data.X
    Y = data.Y

    if diagonal:
        raise NotImplementedError()
    else:
        K_xs = prior.blocks_var(XS,  group_size, block_size)


    K_xx = prior.full_covar(X, X)
    K_xs_x = prior.full_covar(XS, X)

    lik_var = likelihood.full_variance 

    mean_x = prior.vec_mean(X)
    mean_xs = prior.vec_mean(XS)

    Y_vec = stack_rows(Y)

    mu, var = gaussian_prediction_blocks(group_size, block_size, Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)


    return mu, var


@dispatch(Data, 'BatchGP', 'BlockGaussianProductLikelihood', 'DataLatentPermutation')
def predict_blocks(XS, data, group_size, block_size, gp, likelihood, prior, diagonal):
    X = data.X
    Y = data.Y

    if diagonal:
        raise NotImplementedError()
    else:
        K_xs = prior.blocks_var(XS,  group_size, block_size)


    K_xx = prior.full_covar(X, X)
    K_xs_x = prior.full_covar(XS, X)

    # Get liklihood
    # TODO: fix this
    lik_var = jax.scipy.linalg.block_diag(
        *likelihood.likelihood_arr[0].variance
    )

    mean_x = prior.vec_mean(X)
    mean_xs = prior.vec_mean(XS)

    Y_vec = stack_rows(Y)

    mu, var = gaussian_prediction_blocks(group_size, block_size, Y_vec, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    return mu, var

@dispatch(Data, 'BatchGP', 'BlockGaussianProductLikelihood', 'Independent')
def predict_blocks(XS, data, group_size, block_size, gp, likelihood, prior, diagonal):
    X = data.X
    Y = data.Y

    if diagonal:
        raise NotImplementedError()

    if group_size == 1 and block_size == 1:
        K_xs = prior.vec_var(XS)[0]
    elif group_size==1 and block_size == XS.shape[0]:
        K_xs = prior.full_covar(XS, XS)[0]
    else:
        raise RuntimeError()

    K_xx = prior.covar(X, X)[0]
    K_xs_x = prior.covar(XS, X)[0]

    # Get liklihood
    # TODO: fix this
    lik_var = likelihood.likelihood_arr[0].full_variance

    mean_x = prior.vec_mean(X)
    mean_xs = prior.vec_mean(XS)

    mu, var = gaussian_prediction_blocks(group_size, block_size, Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var)

    return mu, var

# =========================== Entry Point For Variational Predictions ===========================

@dispatch(Likelihood, Transform, ApproximatePosterior, True)
@dispatch(Likelihood, Transform, ApproximatePosterior, False)
def predict(XS, data, likelihood, prior, approximate_posterior, inference, whiten, diagonal):
    return  evoke('marginal_prediction', approximate_posterior, likelihood, prior, whiten=whiten)(
        XS, data, approximate_posterior, likelihood, prior, inference, diagonal, whiten
    )
