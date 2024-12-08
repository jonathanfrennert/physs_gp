from ...import settings
from ...kernels import Kernel, RBF
from ...likelihood import Gaussian, GaussianParameterised, ProductLikelihood
from ...approximate_posteriors import GaussianApproximatePosterior, MeanFieldApproximatePosterior
from ...dispatch import dispatch, evoke
from ..gaussian import log_gaussian
from ...transforms import Independent, LinearTransform

from ...utils import utils
from ...utils.utils import can_batch, get_batch_type
from ...utils.nan_utils import mask_to_identity, get_mask, mask_vector, get_diag_mask

from ..matrix_ops import cholesky, log_chol_matrix_det, add_jitter, cholesky_solve, vec_columns, triangular_solve, block_diagonal_from_cholesky, block_from_vec, v_get_block_diagonal, block_from_mat

import jax
from jax import jit
from functools import partial
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList
from batchjax import batch_or_loop, BatchType

@jit
def gaussian_predictive_mean(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    mask = get_mask(Y)
    M = get_diag_mask(mask)
    Y = mask_vector(Y, mask)

    Ns = K_xs.shape[0]
    N = Y.shape[0]

    k = M @ K_xx @ M + lik_var
    k_chol = cholesky(k)

    mu = K_xs_x @ M @ cholesky_solve(k_chol, Y-mean_x) + mean_xs
    mu = np.reshape(mu, [Ns, 1])

    return mu

@jit
def gaussian_predictive_covar(Y, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs, lik_var):
    mask = get_mask(Y)
    M = get_diag_mask(mask)

    Ns_1 = K_xs.shape[0]
    Ns_2 = K_xs.shape[1]
    N = Y.shape[0]

    k = M @ K_xx @ M + lik_var
    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, M @ K_xs_x.T, lower=True)
    A2 = jax.scipy.linalg.solve_triangular(k_chol, M @ K_x_xs, lower=True)
    sig = K_xs - A1.T @ A2

    sig = np.reshape(sig, [Ns_1, Ns_2])

    return sig


@jit
def gaussian_prediction(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    chex.assert_rank(
        [Y, mean_x, mean_xs, K_xs, K_xs_x, K_xx], 
        [2, 2, 2 ,2, 2, 2]
    )

    mask = get_mask(Y)
    M = get_diag_mask(mask)
    Y = mask_vector(Y, mask)

    Ns = K_xs.shape[0]

    k = M @ K_xx @ M + lik_var

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, M @ K_xs_x.T, lower=True)

    mu = K_xs_x @ M @ cholesky_solve(k_chol, Y-mean_x) + mean_xs
    sig = K_xs - A1.T @ A1

    mu = np.reshape(mu, [Ns, 1])
    sig = np.reshape(sig, [Ns, Ns])

    return mu, sig

@jit 
def gaussian_prediction_with_additive_noise_precision(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, sigma_inv):
    mask = get_mask(Y)
    M = get_diag_mask(mask)
    Y = mask_vector(Y, mask)

    Ns = K_xs.shape[0]

    K_m = M.T @ K_xx @ M

    # TODO: this might need to be symetrissed :( 
    A = K_m @ sigma_inv + np.eye(K_m.shape[0])
    A_chol = cholesky(A)

    # TODO: figure out M transpose here
    mu = K_xs_x @ M.T @ sigma_inv @  cholesky_solve(A_chol, Y-mean_x) + mean_xs

    A1 = K_xs_x @ M.T @ sigma_inv
    A2 = cholesky_solve(A_chol, M @ K_xs_x.T)

    sig = K_xs - A1 @ A2

    mu = np.reshape(mu, [Ns, 1])
    sig = np.reshape(sig, [Ns, Ns])

    return mu, sig



@jit 
def gaussian_prediction_diagonal(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    mask = get_mask(Y)
    M = get_diag_mask(mask)
    Y = mask_vector(Y, mask)

    k = M.T @ K_xx @ M + lik_var

    k_chol = cholesky(k)

    A1 = jax.scipy.linalg.solve_triangular(k_chol, M @ K_xs_x.T, lower=True)

    mu = K_xs_x  @ M @ cholesky_solve(k_chol, Y-mean_x) + mean_xs

    sig = K_xs - np.sum(np.square(A1), axis=0)
    sig = sig[:, None]

    chex.assert_equal(mu.shape, sig.shape)

    return mu, sig

@jit 
def gaussian_prediction_diagonal_with_additive_noise_precision(Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, sigma_inv):
    mask = get_mask(Y)
    M = get_diag_mask(mask)
    Y = mask_vector(Y, mask)

    K_m = M.T @ K_xx @ M

    # TODO: this might need to be symetrissed :( 
    sigma_inv_chol = cholesky(add_jitter(sigma_inv, settings.jitter))

    A = K_m @ sigma_inv + np.eye(K_m.shape[0])
    A_chol = cholesky(A)

    # TODO: figure out M transpose here
    mu = K_xs_x @ M.T @ sigma_inv @  cholesky_solve(A_chol, Y-mean_x) + mean_xs

    A1 = K_xs_x @ M.T @ sigma_inv
    A2 = cholesky_solve(A_chol, M @ K_xs_x.T)

    sig = K_xs - np.sum(np.multiply(A1.T, A2), axis=0)
    sig = sig[:, None]

    chex.assert_equal(mu.shape, sig.shape)

    return mu, sig

@partial(jit, static_argnums=(0, 1))
def gaussian_prediction_blocks(group_size, block_size, Y, K_xs, K_xs_x, K_xx, mean_x, mean_xs, lik_var):
    """
    We want to compute
        blk_diag(Kxz Kzz^{-1} m), 
        blk_diag(Kxx - Kxz Kzz^{-1} Kzx)

    The first term only takes O(NM + M^3) so we do not worry about it. For the variance
        we want to avoid forming any O(N^2) matrices so we compute it as:

        blk_diag(Kxx) - blk_diag(Kxz Kzz^{-1} Kzx) =
        blk_diag(Kxx) - blk_diag_cholesky_product(Kxz Kzz^{-1/2})

    where blk_diag(Kxx) is passed through as K_xs

    # TODO: what is group size meant to do, and assert that K_xs is of block size
    """
    N = K_xs.shape[0]
    M = K_xs_x.shape[1]
    Q = block_size


    chex.assert_shape(K_xx, lik_var.shape)

    # TODO: add means and missing data
    K = K_xx + lik_var
    K_chol = cholesky(K)

    #K_xs[0] - K_xs_x @ cholesky_solve(K_chol, K_xs_x.T)
    A = triangular_solve(
        K_chol, 
        K_xs_x.T, 
        lower=True
    )
    chex.assert_shape(A, [M, N*Q])

    B = block_diagonal_from_cholesky(A.T, block_size)
    chex.assert_shape(B, [N, Q, Q])
    chex.assert_shape(K_xs, [N, Q, Q])

    sig = K_xs - B
    chex.assert_shape(sig, [N, Q, Q])

    # K_xs_x @ cholesky_solve(K_chol, Y)
    mu = triangular_solve(K_chol.T, A, lower=False).T @ Y
    mu = block_from_vec(mu, block_size)
    chex.assert_shape(mu, [N, Q])

    return mu, sig
