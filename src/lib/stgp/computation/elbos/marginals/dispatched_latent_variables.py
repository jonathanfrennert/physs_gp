import chex
import jax
import jax.numpy as np
import objax

from ....dispatch import dispatch, evoke
from .... import settings
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_covar, whitened_gaussian_conditional_diagional, whitened_gaussian_conditional_full, gaussian_conditional_blocks, whitened_gaussian_conditional_full , whitened_gaussian_covar
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT, cholesky_solve, triangular_solve, batched_block_diagional, hessian
from ...permutations import left_permute_mat, data_order_to_output_order, permute_vec, permute_mat, unpermute_vec, unpermute_mat
from ....core import Block, get_block_dim

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform, Aggregate
from ....transforms.pdes import DifferentialOperatorJoint
from ....transforms import JointDataLatentPermutation, IndependentDataLatentPermutation, DataLatentPermutation
from ....transforms.latent_variable import LatentVariable, UncertainInput
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian, ConjugateApproximatePosterior, FullConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ....core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part, get_permutated_prior

from .linear_marginals import linear_marginal_blocks

# ==== LV part ====

def latent_variable_marginal(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool, predict: bool, diagonal: bool = True, XS_2 = None):
    # GP prior for which we are adding LV to
    gp_prior = prior.parent.parent[0]
    lv_prior = prior.parent.parent[1]

    gp_q = approximate_posterior.approx_posteriors[0]
    lv_q = approximate_posterior.approx_posteriors[1]

    # only support single GPs atm
    assert type(gp_prior).__name__ == 'GPPrior'
    assert type(lv_prior).__name__ == 'GPPrior'

    # collect data and sparisty

    X = data.X
    Z_gp = gp_prior.sparsity.Z
    Z_lv = lv_prior.sparsity.Z

    # predict the latent variable
    gp_mu = q_m[0]
    gp_S_chol = q_S_chol[0][0]

    lv_mu = q_m[1]
    lv_S_chol = q_S_chol[1][0]
    lv_S = lv_S_chol @ lv_S_chol.T

    if predict:
        lv_pred_mu, lv_pred_var = evoke('marginal_prediction_blocks', lv_q, likelihood, lv_prior, lv_prior.sparsity, whiten=whiten)(
            XS, lv_prior.sparsity, lv_mu[..., None], lv_S_chol[None, None, ...], lv_q, likelihood, lv_prior, lv_prior.sparsity, None, whiten
        )
        lv_pred_mu = lv_pred_mu[None, ..., 0]
        lv_pred_var = lv_pred_var[None, ..., 0, 0]

        #lv_pred_mu = np.zeros_like(lv_pred_mu)
        #lv_pred_var = np.ones_like(lv_pred_var) 
    else:
        lv_pred_mu, lv_pred_var = lv_mu, lv_S
        lv_pred_mu = lv_pred_mu[None, ...]
        lv_pred_var = np.diag(lv_pred_var)[None, :, None]

    XS_concat = prior.transform_x(XS, lv_pred_mu[0])

    def get_mu_var(XS):
        Kzz = gp_prior.covar(Z_gp, Z_gp)
        Kxz = gp_prior.kernel.K(XS, Z_gp)
        Kxx_var = gp_prior.kernel.K_diag(XS)  

        # compute mu and var
        if whiten:
            mu, var =  whitened_gaussian_conditional_diagional(
                XS, 
                Z_gp, 
                Kzz, 
                Kxz, 
                Kxx_var, 
                gp_mu,
                gp_S_chol
            )
        else:
            mu, var =  gaussian_conditional_diagional(
                XS, 
                Z_gp, 
                Kzz, 
                Kxz, 
                Kxx_var, 
                gp_mu,
                gp_S_chol,
                np.zeros(Z_gp.shape[0])[:, None],
                np.zeros(XS.shape[0])[:, None]
            )

        # if not diagonal compute full covariance
        if not diagonal:
            print('bug here, need to fix')
            if whiten:
                var = whitened_gaussian_covar(
                    XS, 
                    XS_2, 
                    Z_gp, 
                    Kzz,
                    Kxz, 
                    gp_prior.kernel.K(Z_gp, XS_2), 
                    gp_prior.kernel.K(XS, XS_2), 
                    m,
                    S_chol
                )
            else:
                var = gaussian_conditional_covar(
                    XS, 
                    XS_2, 
                    Z_gp, 
                    Kzz,
                    Kxz, 
                    gp_prior.kernel.K(Z_gp, XS_2), 
                    gp_prior.kernel.K(XS, XS_2), 
                    m,
                    S_chol
                )

        return mu, mu**2, var

    mu, mu_squared, var = get_mu_var(XS_concat)

    if not diagonal:
        breakpoint()


    mu = mu

    mu_1_diff = jax.vmap(jax.jacobian(get_mu_var))(XS_concat[:, None, :])
    mu_1_diff = mu_1_diff[0][:, 0, 0, 0, prior.w_index][:, None]

    var = var + (mu_1_diff ** 2) * lv_pred_var[0] 

    if prior.with_hessian:
        hess = jax.vmap(hessian(get_mu_var, 0))(XS_concat[:, None, :])
        mu_hess = hess[0][:, 0, 0, 0, prior.w_index, 0, prior.w_index][:, None]

        # update mean with hessian
        #mu = mu + 0.5 * mu_hess * lv_pred_var[0]

        #update variance with hessian
        var = var + 0.5 * (mu_hess**2) * (lv_pred_var[0]**2)

    mu = mu[:, None, ]
    var = var[ :, None,  None, ...]

    chex.assert_rank([mu, var], [3, 4])

    return mu, var


@dispatch(MeanFieldApproximatePosterior, Likelihood, LatentVariable, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, LatentVariable, whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):
    """
    .
    """
    # use GP predictive equations for approximate_posterior[0] but use the MMK kernel 

    return latent_variable_marginal(
        data.X, data,  q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten, False
    )

@dispatch(MeanFieldApproximatePosterior, Likelihood, LatentVariable, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, LatentVariable, whiten=False)
def marginal_prediction(XS, data, approximate_posterior, likelihood, prior, inference, diagonal, whiten, num_samples=None, posterior=False):

    # use GP predictive equations for approximate_posterior[0] but use the MMK kernel 
    q_m, q_S_chol = evoke('variational_params', approximate_posterior, likelihood, prior.base_prior, whiten)(
        data, approximate_posterior, likelihood, prior.base_prior, whiten
    )


    return latent_variable_marginal(
        XS, data,  q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten, True
    )


@dispatch(MeanFieldApproximatePosterior, Likelihood, LatentVariable, Sparsity, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, LatentVariable, Sparsity, whiten=False)
def marginal_prediction_covar(XS_1, XS_2, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    return latent_variable_marginal(
        XS_1, data,  q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten, True, diagonal=False, XS_2 = XS_2
    )[1]




