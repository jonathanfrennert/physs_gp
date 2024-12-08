import chex
import jax
import jax.numpy as np
import objax
from ....dispatch import dispatch, evoke
from ....utils.batch_utils import batch_over_module_types
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT, cholesky_solve, triangular_solve, batched_block_diagional, to_block_diag
from ....core import Block, get_block_dim

from ....transforms import Independent

def meanfield_marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: Block, whiten: bool, XS=None, sparsity=None, prediction=True):
    """ 
    Helper function to collect marginals across a mean-field approximate posterior
    
    This function only accepts an independent prior as we simply batch over the posterior-prior pairs
    and then combine the results into a block diagonal approximate posterior

    NOTE: we combine into a block diagonal matrix as there is no guarentee that the ELL will 
        decompose across these latents and so for simplicty we just assume that it wont
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    assert type(prior) == Independent 

    latents_arr = prior.parent
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(latents_arr)
    N = data.X.shape[0]

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    whiten_arr = [whiten for q in range(num_latents)]

    # TODO: fix block sizes here
    #out_block_arr = [likelihood_arr[0].block_type for q in range(num_latents)]
    out_block_arr = [out_block for q in range(num_latents)]

    N, Q, LB, _ = q_S_chol.shape
    N, QL, B = q_m.shape
    L = int(QL/Q)

    q_m = np.reshape(q_m, [N, Q, L, B])
    sparsity_arr = sparsity

    
    if not prediction:
        # Compute q(f) for each output
        # add additional dimension to q_m and q_S_chol to ensure rank [3, 4] after batching
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
    else:

        marginal_mu, marginal_var = batch_over_module_types(
            evoke_name = 'marginal_prediction_blocks',
            evoke_params = [],
            module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
            fn_params = [XS, data, q_m, q_S_chol[:, :, None, ...], approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, out_block_arr, whiten_arr],
            fn_axes = [None, None, 1, 1, 0, 0, 0, 0, 0, 0],
            dim = len(latents_arr),
            out_dim  = 2,
            evoke_kwargs = {'whiten': whiten}
        )

    chex.assert_rank([marginal_mu, marginal_var], [4, 5])
    Q1, N1, LB_mu, _ = marginal_mu.shape
    Q, N, _, LB, _ = marginal_var.shape

    chex.assert_equal(LB_mu, LB)
    chex.assert_equal(Q, Q1)
    chex.assert_equal(N, N1)

    # fix shapes
    #  each component will return rank (3, 4). stack into independent (block diagonal)
    # convert to N - (Q - P - B ) - 1
    marginal_mu = np.reshape(
        np.transpose(marginal_mu, [1, 0, 2, 3]), 
        [N, Q*LB, 1]
    )
    # Convert to [N, Q, 1, LB, LB]
    marginal_var = np.transpose(marginal_var, [1, 0, 2, 3, 4]) 
    # Remove reduant dim ->  [N, Q, LB, LB]
    marginal_var = marginal_var[:, :, 0, :, :]
    # compute block diagonal ->  [N,QPB, QPB]
    marginal_var = jax.vmap(to_block_diag)(marginal_var)
    # add back missing dim  ->  [N,1, QPB, QPB]
    marginal_var = marginal_var[:, None, ...]
    
    chex.assert_shape(marginal_mu, [N, Q*LB,  1])
    chex.assert_shape(marginal_var, [N, 1, Q*LB,  Q*LB])

    return marginal_mu, marginal_var

