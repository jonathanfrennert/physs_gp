import jax
from jax import jit
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList

from ...import settings
from ...kernels import Kernel, RBF
from ...likelihood import Gaussian
from ...approximate_posteriors import GaussianApproximatePosterior
from ...dispatch import dispatch
from ..gaussian import log_gaussian
from ... import utils
from ..matrix_ops import cholesky, log_chol_matrix_det, add_jitter, diagonal_from_cholesky



@jit
def gaussian_cholesky_kl( mu_1, covar_chol_1, mu_2, covar_chol_2) -> np.ndarray:
    """
    computes KL[g1, g2]
    where
        g1 = N(mu_1, covar_chol_1 @ covar_chol_1.T)
        g2 = N(mu_2, covar_chol_2 @ covar_chol_2.T)

    """

    log_det_term = log_chol_matrix_det(covar_chol_2) - log_chol_matrix_det(covar_chol_1)

    # see https://github.com/GPflow/GPflow/blob/develop/gpflow/kullback_leiblers.py
    trace_term = np.sum(
        np.square(
            jax.scipy.linalg.solve_triangular(covar_chol_2, covar_chol_1, lower=True)
        )
    )

    err = mu_2 - mu_1

    maha_term = np.sum(
        np.square(jax.scipy.linalg.solve_triangular(covar_chol_2, err, lower=True))
    )

    N = mu_1.shape[0] * 1.0

    return 0.5 * (log_det_term - N + trace_term + maha_term)

@jit
def gaussian_kl( mu_1, covar_1, mu_2, covar_2) -> np.ndarray:
    """
    Computes KL[g1, g2] between two gaussians:

        1/2 [log(|Σ2|/|Σ1|)−d+tr{Σ2^{−1} Σ1}+(𝜇2−𝜇1)^T Σ2^{-1}(𝜇2−𝜇1)]
    """
    covar_chol_1 = cholesky(add_jitter(covar_1, settings.jitter))
    covar_chol_2 = cholesky(add_jitter(covar_2, settings.jitter))
    
    return gaussian_cholesky_kl(mu_1, covar_chol_1, mu_2, covar_chol_2)


@jit
def whitened_gaussian_kl(mu_1, covar_chol_1) -> np.ndarray:
    """
    Assumes that g2 is a standard Gaussian - N(0, I)
    """

    covar_1_diag = diagonal_from_cholesky(covar_chol_1)

    log_det_term = -log_chol_matrix_det(covar_chol_1)

    trace_term = np.sum(covar_1_diag)

    #trace_term = np.trace(np.square(covar_chol_1))

    maha_term = np.sum(np.square(mu_1))

    N = mu_1.shape[0] * 1.0

    return 0.5 * (log_det_term - N + trace_term + maha_term)
