import jax
import jax.numpy as np
import chex
import objax

from batchjax import batch_or_loop
from ..transforms import Independent, LinearTransform
from ..utils.utils import get_batch_type

from typing import List



def get_diagonal_gaussian_likelihood_variances(Y: np.ndarray, likelihood) -> np.ndarray:
    num_latents = Y.shape[1]
    N = Y.shape[0]

    def _compute_lik_variance(N, likelihood):
        return likelihood.variance * np.eye(N)

    var_arr = batch_or_loop(
        _compute_lik_variance,
        [ N, likelihood ],
        [ None, 0 ],
        num_latents,
        1,
        get_batch_type(likelihood)
    )

    var_arr = jax.scipy.linalg.block_diag(*var_arr)

    chex.assert_equal(var_arr.shape[0], N * num_latents)
    chex.assert_equal(var_arr.shape[1], N * num_latents)

    return var_arr

def get_vec_gaussian_likelihood_variances(Y: np.ndarray, likelihood) -> np.ndarray:
    num_latents = Y.shape[1]
    N = Y.shape[0]

    def _compute_lik_variance(N, likelihood):
        return likelihood.variance * np.ones(N)

    var_arr = batch_or_loop(
        _compute_lik_variance,
        [ N, likelihood ],
        [ None, 0 ],
        num_latents,
        1,
        get_batch_type(likelihood)
    )

    return var_arr

