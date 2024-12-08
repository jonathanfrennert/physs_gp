""" Helper functions for batching in variational models """
import jax
import objax
import chex

from batchjax import batch_or_loop
from ...utils.utils import get_batch_type

def _batch_over_prior(prior, fn):
    return batch_or_loop(
        fn,
        [prior.latents],
        [0],
        dim = len(prior.latents),
        out_dim = 1,
        batch_type = get_batch_type(prior.latents)
    )

def prior_Z(prior):
    return _batch_over_prior(
        prior,
        lambda prior: prior.sparsity.Z,
    )

def prior_mean_X(prior, X):
    N = X.shape[0]
    Q = prior.num_latents

    mean_arr = _batch_over_prior(
        prior,
        lambda prior: prior.mean(X)[0],
    )

    chex.assert_shape(mean_arr, [Q, N, 1])
    return mean_arr

def prior_mean_Z(prior):
    K_zz_arr = _batch_over_prior(
        prior,
        lambda prior: prior.mean(prior.sparsity.Z)[0],
    )
    return K_zz_arr

def prior_covar_ZZ(prior):
    K_zz_arr = _batch_over_prior(
        prior,
        lambda prior: prior.kernel.K(prior.sparsity.Z, prior.sparsity.Z),
    )
    return K_zz_arr

def prior_covar_XZ(prior, X):
    N = X.shape[0]
    Q = prior.num_latents
    M = prior.latents[0].sparsity.Z.shape[0]

    K_xz_arr = _batch_over_prior(
        prior,
        lambda prior: prior.kernel.K(X, prior.sparsity.Z),
    )

    chex.assert_shape(K_xz_arr, [Q, N, M])
    return K_xz_arr

def prior_covar_X(prior, X):
    N = X.shape[0]
    Q = prior.num_latents

    K_xx_arr = _batch_over_prior(
        prior,
        lambda prior: prior.kernel.K(X, X),
    )
    chex.assert_shape(K_xx_arr, [Q, N, N])
    return K_xx_arr
