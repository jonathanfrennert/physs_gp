"""Standard Gaussian methods."""

import jax
import jax.numpy as np
from jax import jit
import chex
from .. import settings

from .matrix_ops import cholesky, cholesky_solve, log_chol_matrix_det, add_jitter, solve_with_additive_inverse, force_symmetric
from ..utils.nan_utils import mask_to_identity, get_mask, mask_vector

from .linalg import solve, solve_from_cholesky, log_determinant, log_determinant_from_cholesky

@jit
def log_gaussian_scalar(Y, mu, variance):
    # ensure scalar
    chex.assert_rank(mu, 0)
    chex.assert_rank(Y, 0)
    chex.assert_rank(variance, 0)

    # scalar
    N = 1

    c1 = -0.5 * N * np.log(2 * np.pi) - N * 0.5 * np.log(variance)

    err = Y - mu
    mahal = (err * err) / variance

    ll = c1 - 0.5 * mahal

    chex.assert_rank(ll, 0)

    return ll

@jit
def log_gaussian_diagonal(Y, mu, variance):
    return jax.vmap(
        log_gaussian_scalar, 
        [0, 0, 0]
    )(np.squeeze(Y), np.squeeze(mu), np.squeeze(variance))

@jit
def log_gaussian(Y, mu, sigma):
    """
    Computes 
        log N(Y | m, S) = -(1/2) [N log 2π + log |S| + (Y-m)^T S⁻¹ (Y-m)]
    """
    # ensure matrices
    chex.assert_rank(mu, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(sigma, 2)

    # ensure square matrix
    chex.assert_equal(sigma.shape[0], sigma.shape[1])

    #sigma_chol = cholesky(add_jitter(sigma,  settings.jitter))
    sigma_chol = cholesky(sigma)

    N = Y.shape[0]

    c1 = -0.5 * N * np.log(2 * np.pi) 
    c2 = - 0.5 * log_chol_matrix_det(sigma_chol)
    c = c1+c2

    err = Y - mu
    mahal = err.T @ cholesky_solve(sigma_chol, err)

    ml = c - 0.5 * mahal
    return np.squeeze(ml) 


@jit
def log_gaussian_with_mask(Y, mu, sigma, mask):
    """
    Gaussian of the form log N(Y | m, S)

    Let Y_m, Y_o indicate the missing and observed datapoints then this function computes 
        log N(Y | m, S) = log N(Y_o | m_o, S_o) N(Y_m | m_m, S_m)

    Then the log-marginal likelihood only on Y_o is given by:
        log N(Y | m, S) - log N(Y_m | 0, 1) where Y_m = 0.
    which is equal to
        log N(Y | m, S) + (1/2) N_m log 2π

    We work with both Y_m and Y_o to keep matrix dimensions fixed no matter how many missing observations
        there are.
    """
    # ensure matrices
    chex.assert_rank(mu, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(sigma, 2)
    chex.assert_rank(mask, 1)

    # ensure square matrix
    chex.assert_equal(sigma.shape[0], sigma.shape[1])

    # masking
    Y = np.nan_to_num(Y, nan=0.0)
    sigma = mask_to_identity(sigma, mask)
    mu = mask_vector(mu, mask)

    log_gauss_joint = log_gaussian(Y, mu, sigma)

    N = Y.shape[0]

    N_mask = np.sum(1-mask)

    return log_gauss_joint + 0.5 * (N_mask * np.log(2* np.pi))

@jit
def log_gaussian_with_precision_noise_with_mask(Y, mu, sigma_inv, mask):
    """
    Gaussian of the form N(Y | mu, P^{-1}) where P is the precision matrix

    Let Y_m, Y_o indicate the missing and observed datapoints and S = noise_inv then this function computes 

        log N(Y | m, S) = log N(Y_o | m_o, S_o) N(Y_m | m_m, S_m)

    Then the log-marginal likelihood only on Y_o is given by:
        log N(Y | m, S) - log N(Y_m | 0, 1) where Y_m = 0.
    which is equal to
        log N(Y | m, S) + (1/2) N_m log 2π

    We work with both Y_m and Y_o to keep matrix dimensions fixed no matter how many missing observations
        there are.
    """
    # ensure matrices
    chex.assert_rank(mu, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(sigma_inv, 2)
    chex.assert_rank(mask, 1)

    # ensure square matrix
    chex.assert_equal(sigma_inv.shape[0], sigma_inv.shape[1])

    # masking
    Y = np.nan_to_num(Y, nan=0.0)
    sigma_inv = mask_to_identity(sigma_inv, mask)
    mu = mask_vector(mu, mask)

    sigma_inv_chol = cholesky(sigma_inv + settings.jitter * np.eye(sigma_inv.shape[0]))

    N = Y.shape[0]

    c1 = -0.5 * N * np.log(2 * np.pi) 
    # negative as we are working with previsions
    c2 = - 0.5 * (-1)*log_chol_matrix_det(sigma_inv_chol)
    c = c1+c2

    err = Y - mu
    mahal = err.T @ sigma_inv @ err

    ml = c - 0.5 * mahal

    # MASK is one if non nan, zero is nan
    N_mask = np.sum(1-mask)

    log_n = np.log(np.clip(N_mask, 1.0, None))

    return np.sum(np.squeeze(ml)) + 0.5 * (N_mask * np.log(2* np.pi))

@jit
def log_gaussian_with_additive_precision_noise_with_mask(Y, mu, K, sigma_inv, mask):
    """
    Gaussian of the form N(Y | mu, K + sigma_inv^{-1}) which is computed as 

        N(Y | mu, K + sigma_inv^{-1}) = -(1/2) [N log 2π + log |K + sigma_inv^{-1})| + (Y-m)^T [K + sigma_inv^{-1}]⁻¹ (Y-m)]

    The log det is given by
        log |K + sigma_inv^{-1})| = log|Ksigma_inv + I| - log|sigma_inv|

    Let Y_m, Y_o indicate the missing and observed datapoints and S = K + sigma_inv^{-1} then this function computes 

        log N(Y | m, S) = log N(Y_o | m_o, S_o) N(Y_m | m_m, S_m)

    Then the log-marginal likelihood only on Y_o is given by:
        log N(Y | m, S) - log N(Y_m | 0, 1) where Y_m = 0.
    which is equal to
        log N(Y | m, S) + (1/2) N_m log 2π

    We work with both Y_m and Y_o to keep matrix dimensions fixed no matter how many missing observations
        there are.
    """
    # ensure matrices
    chex.assert_rank(mu, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(K, 2)
    chex.assert_rank(sigma_inv, 2)
    chex.assert_rank(mask, 1)

    # ensure square matrix
    chex.assert_equal(sigma_inv.shape[0], sigma_inv.shape[1])

    # masking
    Y = np.nan_to_num(Y, nan=0.0)
    sigma_inv = mask_to_identity(sigma_inv, mask)
    mu = mask_vector(mu, mask)

    sigma_inv_chol = cholesky(add_jitter(sigma_inv, settings.jitter))
    K_chol = cholesky(add_jitter(K, settings.jitter))

    N = Y.shape[0]

    c1 = -0.5 * N * np.log(2 * np.pi) 
    # negative as we are working with previsions
    T = K @ sigma_inv + np.eye(sigma_inv.shape[0])

    if False:
        c2 = - 0.5 * (
            log_chol_matrix_det(cholesky(T)) + 
            (-1)*log_chol_matrix_det(sigma_inv_chol)
        )
    else:
        # the form of T makes cholesky unstable, so just directly take log determinants

        c2 = - 0.5 * (
            np.linalg.slogdet(T)[1] + 
            (-1)*log_chol_matrix_det(sigma_inv_chol)
        ) 

    c = c1+c2

    err = Y - mu
    mahal = err.T  @ solve_with_additive_inverse(K, sigma_inv,  err)

    ml = c - 0.5 * mahal

    # MASK is one if non nan, zero is nan
    N_mask = np.sum(1-mask)

    log_n = np.log(np.clip(N_mask, 1.0, None))

    return np.sum(np.squeeze(ml)) + 0.5 * (N_mask * np.log(2* np.pi))

@jit
def log_gaussian_with_nans(Y, mu, sigma):
    mask = get_mask(Y)
    return log_gaussian_with_mask(Y, mu, sigma, mask)

@jit
def log_gaussian_with_precision_noise_with_nans(Y, mu, sigma_inv):
    mask = get_mask(Y)
    return log_gaussian_with_precision_noise_with_mask(Y, mu, sigma_inv, mask)

@jit
def log_gaussian_with_additive_precision_noise_with_nans(Y, mu, K, sigma_inv):
    mask = get_mask(Y)
    return log_gaussian_with_additive_precision_noise_with_mask(Y, mu, K, sigma_inv, mask)
