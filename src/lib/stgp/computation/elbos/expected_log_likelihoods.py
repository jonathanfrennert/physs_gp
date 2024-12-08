import jax
from jax import jit
import jax.numpy as np
import objax
import chex
from typing import List
from objax import ModuleList

from ...import settings
from ...core import GPPrior
from ...likelihood import Gaussian
from ...approximate_posteriors import GaussianApproximatePosterior, MM_GaussianInnerLayerApproximatePosterior
from ...dispatch import dispatch
from ..gaussian import log_gaussian, log_gaussian_with_nans, log_gaussian_with_precision_noise_with_nans, log_gaussian_scalar
from ..matrix_ops import add_jitter, cholesky, cholesky_solve
from ... import utils
from ...utils.nan_utils import get_mask, mask_to_identity, mask_vector



@jit
def scalar_gaussian_expected_log_likelihood(X:np.ndarray, Y:np.ndarray, noise:np.ndarray, q_mu:np.ndarray, q_covar_diag:np.ndarray) ->  np.ndarray:
    chex.assert_equal(X.shape[0], 1)
    chex.assert_shape(Y, [1, 1])
    chex.assert_equal(Y.shape, q_mu.shape)
    chex.assert_equal(Y.shape, q_covar_diag.shape)

    N = Y.shape[0]
    c1 = -0.5*np.log(2*np.pi) - 0.5*np.log(noise)

    err = Y - q_mu
    err = np.sum(np.matmul(err.T, err))

    ell =  N*c1  -0.5*(err + np.sum(q_covar_diag))/noise


    chex.assert_rank(ell, 0)
    return ell

@jit
def gaussian_expected_log_likelihood(X:np.ndarray, Y:np.ndarray, noise:np.ndarray, q_mu:np.ndarray, q_covar_diag:np.ndarray) ->  np.ndarray:

    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(Y.shape[1], 1)
    chex.assert_equal(Y.shape, q_mu.shape)
    chex.assert_equal(Y.shape, q_covar_diag.shape)

    N = Y.shape[0]
    c1 = -0.5*np.log(2*np.pi) - 0.5*np.log(noise)

    err = Y - q_mu
    err = np.sum(np.matmul(err.T, err))

    ell =  N*c1  -0.5*(err + np.sum(q_covar_diag))/noise


    chex.assert_rank(ell, 0)
    return ell

@jit
def diagonal_gaussian_expected_log_likelihood(X:np.ndarray, Y:np.ndarray, noise:np.ndarray, q_mu:np.ndarray, q_covar_diag:np.ndarray) ->  np.ndarray:
    """
        Args:
            X: N x D
            Y: N x 1
            noise: N x 1
            q_mu: N x 1
            q_covar_diag: N x 1
    """

    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(Y.shape[1], 1)
    chex.assert_equal(Y.shape, q_mu.shape)
    chex.assert_equal(Y.shape, q_covar_diag.shape)

    N = Y.shape[0]
    c1 = -0.5*N*np.log(2*np.pi) - 0.5*np.sum(np.log(noise))

    err = Y - q_mu
    inv_noise = 1/noise
    err = np.sum(np.matmul(err.T, np.multiply(inv_noise, err)))

    ell =  c1  -0.5*(err + np.sum(np.multiply(inv_noise, q_covar_diag)))

    chex.assert_rank(ell, 0)
    return ell

@jit
def full_gaussian_expected_log_likelihood(X:np.ndarray, Y:np.ndarray, noise:np.ndarray, q_mu:np.ndarray, q_covar:np.ndarray) ->  np.ndarray:
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(q_covar,  2)
    chex.assert_equal(Y.shape[1], 1)
    chex.assert_equal(Y.shape, q_mu.shape)
    chex.assert_shape(q_covar, [Y.shape[0], Y.shape[0]])
    chex.assert_shape(noise, q_covar.shape)


    ml =  log_gaussian_with_nans(Y, q_mu, noise) 

    mask = get_mask(Y)
    N_mask = np.sum(1-mask)

    masked_noise = mask_to_identity(noise, mask)
    masked_noise_chol = cholesky(masked_noise)
    masked_q_covar = mask_to_identity(q_covar, mask)

    traced_term = cholesky_solve(masked_noise_chol, masked_q_covar)

    trace_term = -0.5*np.trace(traced_term)+0.5*N_mask

    ell =  ml + trace_term

    chex.assert_rank(ell, 0)
    return ell

@jit
def full_gaussian_expected_log_precision_likelihood(X:np.ndarray, Y:np.ndarray, noise_precision:np.ndarray, q_mu:np.ndarray, q_covar:np.ndarray) ->  np.ndarray:
    """
    Block expected log likelihood where the likelihood is parameterised by its precision
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_rank(q_covar,  2)
    chex.assert_equal(Y.shape[1], 1)
    chex.assert_equal(Y.shape, q_mu.shape)
    chex.assert_shape(q_covar, [Y.shape[0], Y.shape[0]])
    chex.assert_shape(noise_precision, q_covar.shape)

    ml =  log_gaussian_with_precision_noise_with_nans(Y, q_mu, noise_precision) 

    mask = get_mask(Y)

    masked_q_covar = mask_to_identity(q_covar, mask)

    traced_term = noise_precision @ q_covar
    masked_trace_term = mask_vector(traced_term, mask)

    trace_term = -0.5*np.trace(masked_trace_term)

    ell =  ml + trace_term

    chex.assert_rank(ell, 0)
    return ell


@jit
def scalar_poisson_expected_log_likelihood(X:np.ndarray, Y:np.ndarray,  binsize:float, q_mu:np.ndarray, q_covar_diag:np.ndarray) -> np.ndarray:
    """
        X, Y, q_mu, q_covar are all scalars
        Let a = E[f] = m and b = E[exp(f)] = exp(m+v/2)  and Possion(y|m) = e^{-m} * (m^y) / (y!) then

            E[log Poisson(y | exp(f)*binsize)] = Y log binsize  + E[Y * log exp(f)] - E[binsize * exp(f)] - log Y!
                                               = Y log binsize + Y * m - binsize * exp(m + v/2) - log Y!

    """
    #TODO: this is assuming an exp link function -- assert this 

    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)
    chex.assert_equal(Y.shape[1], 1)
    chex.assert_equal(Y.shape, q_mu.shape)
    chex.assert_equal(Y.shape, q_covar_diag.shape)

    Y = np.squeeze(Y)
    q_mu = np.squeeze(q_mu)
    binsize = np.squeeze(binsize)

    ell =  Y*np.log(binsize) + Y*q_mu - binsize*np.exp(q_mu + q_covar/2) - gammaln(Y+1.0)

    chex.assert_rank(ell, 0)
    return ell


