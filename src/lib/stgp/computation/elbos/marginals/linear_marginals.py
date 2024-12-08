import chex
import jax
import jax.numpy as np
import objax

from ....dispatch import dispatch, evoke
from .... import settings
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_covar, whitened_gaussian_conditional_diagional, whitened_gaussian_conditional_full
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT
from ....core import Block
from ....core.block_types import compare_block_types

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform, Aggregate
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ....core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part, get_linear_model_part_list

def linear_marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, out_block: int, whiten: bool, XS=None, sparsity=None):
    """ Recursively compute the transformed linear marginal. """

    parent_prior = prior.hierarchical_base_prior

    prior_list = get_linear_model_part_list(prior)

    # get linear part input dim
    if prior_list[0].in_block_dim is None:
        # we do not need to worry about the block size
        # so just set in 
        parent_out_block_dim = out_block
    else:
        parent_out_block_dim = compare_block_types(
            out_block, 
            prior_list[0].in_block_type
        )

    # get parent transformed value
    if XS is None:
        mu_parent, var_parent  = evoke('marginal_blocks', approximate_posterior, likelihood, parent_prior, whiten=whiten)(
            data, q_m, q_S, approximate_posterior, likelihood, parent_prior, parent_out_block_dim, whiten
        ) 
    else:
        if type(sparsity) is not list:
            sparsity = [sparsity]

        mu_parent, var_parent  = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, parent_prior, sparsity[0], whiten=whiten, debug=False)(
            XS, data, q_m, q_S, approximate_posterior, likelihood, parent_prior, sparsity, parent_out_block_dim, whiten
        ) 

    chex.assert_rank([mu_parent, var_parent], [3, 4])

    if prior.out_block_dim != None:
        breakpoint()

    # Check if we can use a diagonal transform or not
    if (var_parent.shape[-1] == 1) or (var_parent.shape[-1] == 1 and out_block == Block.DIAGONAL):
        chex.assert_rank([mu_parent, var_parent], [3, 4])

        # mu_parent, var_parent are the mean and variance of the base GP.
        # we now transform them through the linear prior
        mu_p, var_p = mu_parent, var_parent
        for p in prior_list:
            # We batch over N, so the transform should support mu of rank 2, and var of rank 3
            if p.full_transform:
                mu_p, var_p = p.transform_diagonal(mu_p, var_p, data)
            else:
                mu_p, var_p = jax.vmap(p.transform_diagonal, [0, 0])(mu_p, var_p)

    elif var_parent.shape[-1] > 1:

        # mu_parent, var_parent are the mean and variance of the base GP.
        # we now transform them through the linear prior

        mu_p, var_p = mu_parent, var_parent
        for p in prior_list:
            if p.full_transform:
                mu_p, var_p = p.transform(mu_p, var_p, data)
            else:
                mu_p, var_p = jax.vmap(p.transform, [0, 0])(mu_p, var_p)
    else:
        breakpoint()
        raise NotImplementedError()

    return mu_p, var_p
