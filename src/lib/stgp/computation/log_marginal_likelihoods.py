# Import Types
from ..data import Data, TransformedData
from ..kernels import Kernel, RBF
from ..likelihood import Likelihood, Gaussian, GaussianParameterised, ProductLikelihood, GaussianProductLikelihood, BlockDiagonalGaussian, DiagonalGaussian, PrecisionBlockDiagonalGaussian
from ..dispatch import dispatch, evoke
from ..utils.batch_utils import batch_over_module_types
from .gaussian import log_gaussian, log_gaussian_with_nans, log_gaussian_with_additive_precision_noise_with_nans
from ..transforms import Independent, LinearTransform, Transform, Joint
from .model_ops import get_diagonal_gaussian_likelihood_variances
from .matrix_ops import vec_columns, stack_rows, to_block_diag
from ..models import BatchGP
from ..utils import utils
from ..utils.utils import get_batch_type
from ..utils.nan_utils import get_same_shape_mask
from .permutations import data_order_to_output_order, permute_mat_ld_to_dl, permute_mat_dl_to_ld, permute_vec_dl_to_ld, permute_mat_tps_to_tsp
from ..core.models import Model
from ..core.model_types import get_model_type, LinearModel, NonLinearModel

from ..dispatch import _ensure_str

from ..utils.nan_utils import mask_to_identity, get_mask, mask_vector
from ..utils.utils import can_batch

import jax
import jax.numpy as np
from jax import jit
import chex
from typing import List
import objax
from objax import ModuleList
from typing import List
from batchjax import batch_or_loop, BatchType

# =================================== Individual Likelihoods ===================================

@dispatch(DiagonalGaussian)
def log_marginal_likelihood(
        X: np.ndarray, Y: np.ndarray, likelihood: Gaussian, K: np.ndarray, mean: np.ndarray
):
    """
    Log marginal likelihood of GP prior with Gaussian likelihood.

    Computes:
        log N(Y | 0, K(X, X) + lik.variance*I)

    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 2)

    N = X.shape[0]

    chex.assert_shape(Y, [N, 1])
    chex.assert_shape(mean, [N, 1])
    chex.assert_shape(K, [N, N])

    lik_noise = likelihood.full_variance

    k = K + lik_noise 

    return log_gaussian_with_nans(Y, mean, k) 

@dispatch(PrecisionBlockDiagonalGaussian)
def log_marginal_likelihood(
        X: np.ndarray, Y: np.ndarray, likelihood: PrecisionBlockDiagonalGaussian, K: np.ndarray, mean: np.ndarray
):
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 3)

    # Y is in time - latent - space format
    # convert to time - latent format
    Y = np.reshape(Y, [-1, 1])

    N = X.shape[0]

    lik_precison = likelihood.full_precision

    return log_gaussian_with_additive_precision_noise_with_nans(Y, mean, K, lik_precison) 


@dispatch(BlockDiagonalGaussian)
def log_marginal_likelihood(
        X: np.ndarray, Y: np.ndarray, likelihood: BlockDiagonalGaussian, K: np.ndarray, mean: np.ndarray
):
    """
    Log marginal likelihood of GP prior with Gaussian likelihood.

    Computes:
        log N(Y | 0, K(X, X) + lik.variance*I)

    """
    chex.assert_rank(X, 2)
    chex.assert_rank(Y, 3)

    # Y is in time - latent - space format
    # convert to time - latent format
    Y = np.reshape(Y, [-1, 1])

    N = X.shape[0]

    lik_noise = likelihood.full_variance

    k = K + lik_noise 

    return log_gaussian_with_nans(Y, mean, k) 



@dispatch(Gaussian)
def log_marginal_likelihood(
        X: np.ndarray, Y: np.ndarray, likelihood: Gaussian, K: np.ndarray, mean: np.ndarray
):
    """
    Log marginal likelihood of GP prior with Gaussian likelihood.

    Computes:
        log N(Y | 0, K(X, X) + lik.variance*I)

    """
    chex.assert_rank(Y, 2)

    N = X.shape[0]

    chex.assert_shape(Y, [N, 1])
    chex.assert_shape(mean, [N, 1])
    chex.assert_shape(K, [N, N])

    lik_noise = likelihood.full_variance

    k = K + lik_noise * np.eye(N)

    return log_gaussian_with_nans(Y, mean, k) 

# =================================== Multioutput Models ===================================


@dispatch(PrecisionBlockDiagonalGaussian, Transform)
@dispatch(BlockDiagonalGaussian, Transform)
def log_marginal_likelihood(
        data, gp: 'Posterior', likelihood: BlockDiagonalGaussian, prior: Transform
):
    """ 
    Prior is a multi-output prior and likelihood is block diagional 
    
    Prior is defined in latent-data format and likelihood is defined in data-latent format (ie one block per datapoint).
    To compute the lml we convert the likelihood to latent-data format.
    """
    X = data.X
    Y = data.Y

    chex.assert_rank(Y, 2)
    _, P = Y.shape
    N = X.shape[0]

    chex.assert_equal(P, likelihood.block_size)

    # Convert Y to latent-data format
    Y_vec = permute_vec_dl_to_ld(np.reshape(Y, [-1, 1]), likelihood.num_latents, N) 

    # precompute prior covariance
    # in latent-data format
    k_xx_arr = prior.covar(X, X)
    mean_arr = prior.mean(X) 

    # in data-latent format
    if _ensure_str(likelihood) == 'BlockDiagonalGaussian':
        likelihood_mat = likelihood.variance
    elif _ensure_str(likelihood) == 'PrecisionBlockDiagonalGaussian':
        likelihood_mat = likelihood.precision

    # Permute so that the ordering between likelihood_mat and Y is the same
    N = X.shape[0]
    NS = likelihood_mat.shape[0]

    lik_mat_bd_tsp = permute_mat_tps_to_tsp(likelihood_mat, likelihood.num_latents)
    lik_mat_tsp = to_block_diag(lik_mat_bd_tsp)
    # convert to latent-data
    ordered_likelihood_mat = permute_mat_dl_to_ld(lik_mat_tsp, likelihood.num_latents, N)

    chex.assert_shape(Y_vec, mean_arr.shape)
    chex.assert_shape(k_xx_arr, ordered_likelihood_mat.shape)

    if _ensure_str(likelihood) == 'BlockDiagonalGaussian':
        return log_gaussian_with_nans(
            Y_vec,
            mean_arr,
            k_xx_arr + ordered_likelihood_mat
        )
    elif _ensure_str(likelihood) == 'PrecisionBlockDiagonalGaussian':
        return log_gaussian_with_additive_precision_noise_with_nans(
            Y_vec,
            mean_arr,
            k_xx_arr,
            ordered_likelihood_mat
        )

@dispatch(ProductLikelihood, Joint)
@dispatch(ProductLikelihood, LinearTransform)
def log_marginal_likelihood(
        data, gp: 'Posterior', likelihood: ProductLikelihood, prior: Independent
):
    """ Independent Latent functions. Each marginal liklihood is computed separately and summed """

    X = data.X
    Y = data.Y

    N, P = Y.shape

    num_outputs = prior.output_dim

    # precompute prior covariance in latent-data format
    Kxx = prior.covar(X, X)
    mean = prior.mean(X) 

    # put Y in latent-data format
    Y_vec = Y.reshape([-1], order='F')[..., None]

    # construct Likelihood

    # get likelihood in data-latent format
    lik_var = likelihood.variance
    lik_var = np.tile(lik_var, [N,  1])

    # convert to latent-data format
    lik_var_vec = lik_var.reshape([-1], order='F')
    lik_var_diag = np.diag(lik_var_vec)

    return log_gaussian_with_nans(Y_vec, mean, Kxx + lik_var_diag)


@dispatch(ProductLikelihood, Independent)
def log_marginal_likelihood(
        data, gp: 'Posterior', likelihood: ProductLikelihood, prior: Independent
):
    """ Independent Latent functions. Each marginal liklihood is computed separately and summed """

    X = data.X
    Y = data.Y

    num_outputs = prior.output_dim

    # precompute prior covariance
    k_xx_arr = prior.covar_blocks(X, X)
    mean_arr = prior.mean_blocks(X) 

    chex.assert_rank(k_xx_arr, 3)
    chex.assert_rank(mean_arr, 3)

    # Ensure batched Y has rank 2
    Y = Y[..., None]

    likelihood_arr = likelihood.likelihood_arr

    # Compute lml for each likelihood and prior
    lml_arr = batch_over_module_types(
        evoke_name = 'log_marginal_likelihood',
        evoke_params = [],
        module_arr = [likelihood_arr],
        fn_params = [X, Y, likelihood_arr, k_xx_arr, mean_arr],
        fn_axes = [None, 1, 0, 0, 0],
        dim = num_outputs,
        out_dim  = 1 
    )
    lml_arr = np.array(lml_arr)
    #chex.assert_shape(lml_arr, (num_outputs, ))

    lml =  np.sum(lml_arr)

    return lml


# ===============================================================================================
# ===============================================================================================
# ========================================  ENTRY POINTs ========================================
# ===============================================================================================
# ===============================================================================================


# ====================================== Linear Transforms ======================================

@dispatch(Data, Model, Likelihood, LinearModel)
def log_marginal_likelihood( data, m, likelihood, prior):

    # dispatch on likelihood and prior
    return  evoke('log_marginal_likelihood', likelihood, prior)(
        data, m, likelihood, prior 
    )


# ====================================== NonLinear Models ======================================
@dispatch(Data, Model, Likelihood, NonLinearModel)
def log_marginal_likelihood( data, gp, likelihood, prior):
    raise RuntimeError('Batch Inference is not supported for Nonlinear Models. Try using Variational inference instead.')

# ======================================  Models ======================================

@dispatch(Data, Model, Likelihood, Transform)
def log_marginal_likelihood( data, m, likelihood, prior):

    model_type = get_model_type(prior)

    return  evoke('log_marginal_likelihood', data, m, likelihood, model_type)(
        data, m, likelihood, prior 
    )


@dispatch(TransformedData, Model, Likelihood, Transform)
def log_marginal_likelihood( data, m, likelihood, prior):
    """
    Let Y = T(A) then
        log p(Y | f) = log p(T^{-1}(Y) | f) + log |dT/dY|
    """

    model_type = get_model_type(prior)

    ll =   evoke('log_marginal_likelihood', data, m, likelihood, model_type)(
        data, m, likelihood, prior 
    )

    return ll + np.sum(data.log_jacobian(data.Y_base))
