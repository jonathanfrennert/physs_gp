import chex
import jax
import jax.numpy as np
import objax

from ....dispatch import dispatch, evoke

from ....approximate_posteriors import  MeanFieldApproximatePosterior
from ....likelihood import Likelihood
from ....transforms import PrecomputedNearestNeighbours, Independent
from ....utils.batch_utils import batch_over_module_types
from ....core import Block, get_block_dim

from ...matrix_ops import to_block_diag


@dispatch(MeanFieldApproximatePosterior, Likelihood, PrecomputedNearestNeighbours, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, PrecomputedNearestNeighbours, whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):
    """
    Args:
        q_m: Q x M x B
        q_S_chol: 1 x  Q  x MB x MB
    """

    # for each group compute the required marginals

    assert type(prior.parent) == Independent 
    ind_prior = prior.parent

    out_block: Block = likelihood.block_type

    latents_arr = ind_prior.parent
    num_latents = len(latents_arr)


    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = ind_prior.get_sparsity_list()

    whiten_arr = [whiten for q in range(num_latents)]
    likelihood_arr = [likelihood for q in range(num_latents)] # likelihood is shared
    out_block_arr = [out_block for q in range(num_latents)] # likelihood is shared

    # TODO: double check all of this
    # TODO: what is L again?
    N, Q, LB, _ = q_S_chol.shape
    N, QL, B = q_m.shape
    L = int(QL/Q)

    q_m = np.reshape(q_m, [N, Q, L, B])
    q_S_chol = np.transpose(q_S_chol, [1, 0, 2, 3])


    q_S_chol_bd = to_block_diag(q_S_chol[:, 0, ...])
    q_m_bd =  np.reshape(q_m, [N*Q, L, B])

    breakpoint()

    base_data = data.base_data
    q_m_groups = q_m_bd[data.neighbour_arr]
    #q_S_chol_bd[data.neighbour_arr, ...][..., data.neighbour_arr]
    q_S_chol_groups = jax.vmap(lambda idx: q_S_chol_bd[idx, ...][..., idx])(data.neighbour_arr)

    breakpoint()


    if True:
        XS = data.X
        XS_groups = np.reshape(XS, [10, 10, -1])

        marginal_mu, marginal_var = batch_over_module_types(
            evoke_name = 'marginal_prediction_blocks',
            evoke_params = [],
            module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
            fn_params = [XS_groups, data, q_m, q_S_chol[:, :, None, ...], approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, out_block_arr, whiten_arr],
            fn_axes = [0, None, 1, 1, 0, 0, 0, 0, 0, 0],
            dim = len(latents_arr),
            out_dim  = 2,
            evoke_kwargs = {'whiten': whiten}
        )

        marginal_mu = np.reshape(marginal_mu, [XS.shape[0], 1, 1])
        marginal_var = np.reshape(marginal_var, [XS.shape[0], 1, 1, 1])

        return marginal_mu, marginal_var
    else:
        marginal_mu, marginal_var = batch_over_module_types(
            evoke_name = 'marginal_blocks',
            evoke_params = [],
            module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr],
            fn_params = [data, q_m, q_S_chol[:, :, None, ...], approx_posteriors_arr, likelihood_arr, latents_arr, out_block_arr, whiten_arr],
            fn_axes = [None, 1, 1, 0, 0, 0, 0, 0],
            dim = len(latents_arr),
            out_dim  = 2,
            evoke_kwargs = {'whiten': whiten}
        )


    # stack results
    breakpoint()

@dispatch(MeanFieldApproximatePosterior, Likelihood, PrecomputedNearestNeighbours, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, PrecomputedNearestNeighbours, whiten=False)
def marginal_prediction(XS, data, approximate_posterior, likelihood, prior, inference, diagonal, whiten, num_samples=None, posterior=False):

    # compute nearest neighbours for XS or use precomputed ones

    # for each group compute the required marginals

    # stack results

    breakpoint()
