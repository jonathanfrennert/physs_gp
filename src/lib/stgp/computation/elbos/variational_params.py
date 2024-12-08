"""
Collects the required parts of q(u) for a given ELBO.

By convention in the single output / diagonal settings the output will be:
    mu: M x 1
    var: M x 1

Let B = the block size, N_b the number of blocks then In the multi-output/block diagonal setting: 
    mu: N_b x B*Q x 1
    var: N_b x B*Q x B*Q

When required we enforce these conventations through assertions.

The way the variational parameters are returns also depends on the type of approximate posterior.

    - In the Meanfield/FullPosterior case we return the mean and cholesky of the variance.

    - For a ConjugateApproximatePosterior we only return the required blocks for computed the ELL terms.

"""
import chex
import jax
import jax.numpy as np

from ...dispatch import dispatch, evoke
from ...utils.batch_utils import batch_over_module_types
from ..marginals import gaussian_conditional_diagional, gaussian_conditional
from ..matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, to_block_diag

# Import Types
from ...transforms import Transform, LinearTransform, Independent, NonLinearTransform, DataStack
from ...transforms.latent_variable import LatentVariable
from ...approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, FullConjugateGaussian, MeanFieldAcrossDataApproximatePosterior
from ...likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood, Gaussian
from ...sparsity import FreeSparsity, Sparsity

# ================================== Dispatched q(u) ==============================
# 
# Must return ranks [2, 3]
# these are for individual gps so there are no latents dimension needed. This will 
# be added in the entry points

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'NoSparsity', False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'NoSparsity', True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'FullSparsity', False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'FullSparsity', True)
@dispatch('FullGaussianApproximatePosterior', Likelihood, Transform, 'SpatialSparsity', False)
@dispatch('FullGaussianApproximatePosterior', Likelihood, Transform, 'SpatialSparsity', True)
@dispatch('DiagonalGaussianApproximatePosterior', Likelihood, 'GPPrior', 'FullSparsity', False)
@dispatch('DiagonalGaussianApproximatePosterior', Likelihood, 'GPPrior', 'FullSparsity', True)
@dispatch('DiagonalGaussianApproximatePosterior', Likelihood, 'GPPrior', 'NoSparsity', True)
@dispatch('DiagonalGaussianApproximatePosterior', Likelihood, 'GPPrior', 'NoSparsity', False)
@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', 'FullSparsity', False)
@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', 'FullSparsity', True)
@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', 'NoSparsity', True)
@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', 'NoSparsity', False)
@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', 'SpatialSparsity', True)
@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', 'SpatialSparsity', False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
    """ For computational reasons we return S_chol """

    # M x 1, M x M
    mu, var_chol =  approximate_posterior.m, approximate_posterior.S_chol

    #add missing dimension
    var_chol = var_chol[None, ...]

    chex.assert_rank([mu, var_chol], [2, 3])
    return mu, var_chol


@dispatch('ConjugatePrecisionGaussian', Likelihood, 'GPPrior', 'NoSparsity', False)
@dispatch('ConjugateGaussian', Likelihood, 'GPPrior', 'NoSparsity', False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
    """
    With no sparsity there is no conditional we need to integrate through in the 
        approxiamte posterior so we directly return the maginal predictions at training data points 

    This is only valid in the single latent function case

    output:
        mu: Mx1
        var: Mx1x1
    """
    mu, var = approximate_posterior.surrogate.posterior(diagonal=True)
    N = mu.shape[0]

    mu = np.reshape(mu, [N, 1])
    var = np.reshape(var, [N, 1, 1])

    chex.assert_rank([mu, var], [2, 3])
    return mu, var

@dispatch('ConjugatePrecisionGaussian', Gaussian, 'GPPrior', 'SpatialSparsity', False)
@dispatch('ConjugatePrecisionGaussian', DiagonalLikelihood, 'GPPrior', 'SpatialSparsity', False)
@dispatch('ConjugateGaussian', Gaussian, 'GPPrior', 'SpatialSparsity', False)
@dispatch('ConjugateGaussian', DiagonalLikelihood, 'GPPrior', 'SpatialSparsity', False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
    """
    output:
        mu: NtxNs
        var: NtxNsxNs
    """
    mu, var = approximate_posterior.surrogate.posterior_blocks()
    Nt = mu.shape[0]
    Ns = mu.shape[1]

    mu = np.reshape(mu, [Nt, Ns])
    var = np.reshape(var, [Nt, Ns, Ns])

    chex.assert_rank([mu, var], [2, 3])
    return mu, var

@dispatch('ConjugatePrecisionGaussian', BlockDiagonalLikelihood, 'GPPrior', Sparsity, False)
@dispatch('ConjugatePrecisionGaussian', BlockDiagonalLikelihood, 'GPPrior', 'NoSparsity', False)
@dispatch('ConjugateGaussian', BlockDiagonalLikelihood, 'GPPrior', Sparsity, False)
@dispatch('ConjugateGaussian', BlockDiagonalLikelihood, 'GPPrior', 'NoSparsity', False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
    """
    Let B = the block size, N_b the number of blocks then
    output:
        mu: N_b x B x 1
        var: N_b x B x B
    """

    block_size = likelihood.block_size

    mu, var = approximate_posterior.surrogate.posterior_blocks()

    # Normalize shapes
    mu = np.reshape(mu, [-1, block_size])
    var = np.reshape(var, [-1, block_size, block_size])

    chex.assert_rank([mu, var], [2, 3])
    return mu, var

#================== SINGLE GP ENTRY POINT ==========================
@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', False)
@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', True)
def variational_params(data, approximate_posterior, likelihood, prior, whiten):
    """  Single approximate posterior setting """
    sparsity = prior.sparsity

    mu, var = evoke('variational_params', approximate_posterior, likelihood, prior, sparsity, whiten)(
        data, approximate_posterior, likelihood, prior, sparsity, whiten
    ) 
    chex.assert_rank([mu, var], [2, 3])


    # there is only one latent gp so just add the single latent missing dimensions 
    mu = mu[:, :, None]
    var = var[:, None, ...]

    chex.assert_rank([mu, var], [3, 4])
    return mu, var

@dispatch(MeanFieldAcrossDataApproximatePosterior, Likelihood, DataStack, Sparsity, False)
@dispatch(MeanFieldAcrossDataApproximatePosterior, Likelihood, DataStack, Sparsity, True)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
    """
    TODO: this is quite an inefficient way of implementing MeanFieldAcrossDataApproximatePosteriors
     but it is useful for a first implementation and for debugging
    """
    sparsity_arr = prior.get_sparsity_list()[0]
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    latents_arr = prior.parent
    num_latents = len(latents_arr)

    whiten_arr = [whiten for q in range(num_latents)]
    likelihood_arr = [likelihood for q in range(num_latents)]

    q_m, q_S = batch_over_module_types(
        evoke_name = 'variational_params',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, whiten_arr],
        fn_params = [data, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, whiten_arr],
        fn_axes = [None, 0, None, 0, 0, 0],
        dim = num_latents,
        out_dim  = 2
    )
    q_S = to_block_diag(q_S[:, 0, :])[None, ...]
    q_m = np.reshape(q_m, [-1, 1])
    return q_m, q_S

#================== MEAN FIELD  ENTRY POINT ==========================
@dispatch(MeanFieldApproximatePosterior, Likelihood, Independent, False)
@dispatch(MeanFieldApproximatePosterior, Likelihood, Independent, True)
def variational_params(data, approximate_posterior, likelihood, prior, whiten):
    """  
    Mean-field approximate posterior setting. Collect parameters across all components q(u_q)

    There are two cases:
        1) Each component is a single Gaussian q(u_q) (like a GaussianPosterior)
            In this case after batching q_m will have a format like [Q, N, B] which we reorginise to
                [N , Q*B, 1]
        2) Each component is a multivariate Gaussian (like a FullGaussianPosterior)
            In this case q_m may have a format like [Q, Nt, Ns*L*B] where L is the output of each component
            Nt, and Ns are stored separately to handle spatial sparsity.


    """
    base_prior = prior.base_prior

    latents_arr = base_prior.parent
    num_latents = len(latents_arr)
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = base_prior.get_sparsity_list()

    likelihood_arr = likelihood.likelihood_arr

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]

    whiten_arr = [whiten for q in range(num_latents)]

    q_m, q_S = batch_over_module_types(
        evoke_name = 'variational_params',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, whiten_arr],
        fn_params = [data, approx_posteriors_arr, likelihood_arr[0], latents_arr, sparsity_arr, whiten_arr],
        fn_axes = [None, 0, None, 0, 0, 0],
        dim = num_latents,
        out_dim  = 2
    )

    # q_m is [Q x M x B]
    # q_S is [Q x 1 x MB x MB]

    # TODO: fix this
    if False:
        if type(q_m) is list:
            # hack for now to get models with mixed number of inducing points to work
            return q_m, q_S

    if q_S.shape[1] == 1:
        # q_m, q_S are batched across latents in the first dimension. Transpose to make it the last axis

        # convert to [N, Q, B] format
        q_m = np.transpose(q_m, [1, 0, 2])
        q_S = np.transpose(q_S, [1, 0, 2, 3])

        # convert to [N, Q*B, 1] format
        q_m = np.reshape(q_m, [q_m.shape[0], -1, 1])

    else:
        L = base_prior.parent[0].output_dim
        Q, N, LB = q_m.shape

        # Convert to [N, Q, LB] format
        q_m = np.transpose(q_m, [1, 0, 2])

        # Convert to [N, QLB, 1] format
        q_m = np.reshape(q_m, [N, Q*LB, 1])

        # convert to [Nt, Q, QLB, QLB] format
        q_S = np.transpose(q_S, [1, 0, 2, 3])


    chex.assert_rank([q_m, q_S], [3, 4])
    return q_m, q_S

#================== DENSE FULL POSTERIOR ENTRY POINT ==========================
@dispatch(FullConjugateGaussian, Likelihood, Transform, 'NoSparsity', True)
@dispatch(FullConjugateGaussian, Likelihood, Transform, 'NoSparsity', False)
@dispatch(FullConjugateGaussian, Likelihood, Transform, Sparsity, True)
@dispatch(FullConjugateGaussian, Likelihood, Transform, Sparsity, False)
def variational_params(data, approximate_posterior, likelihood, prior, sparsity, whiten):
    """  conjugate Full-posterior approximate posterior setting """
    q_m, q_S =  approximate_posterior.surrogate.posterior_blocks()

    chex.assert_rank([q_m, q_S], [3, 4])

    # must return rank [2, 3] as this is for a single latent function
    q_m = q_m[..., 0]
    q_S = q_S[:, 0, ...]
    return q_m, q_S


@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, True)
def variational_params(data, approximate_posterior, likelihood, prior, whiten):
    """  Full-posterior approximate posterior setting """

    sparsity_arr = prior.base_prior.get_sparsity_list()

    #TODO: assuming same sparsity across all latents
    sparsity = sparsity_arr[0]

    mu, var = evoke('variational_params', approximate_posterior, likelihood, prior, sparsity, whiten)(
        data, approximate_posterior, likelihood, prior, sparsity, whiten
    ) 
    chex.assert_rank([mu, var], [2, 3])

    # add missing dimension
    mu = mu[..., None]
    var = var[:, None, ...]

    chex.assert_rank([mu, var], [3, 4])
    return mu, var

@dispatch(ApproximatePosterior, Likelihood, Transform, False)
@dispatch(ApproximatePosterior, Likelihood, Transform, True)
def variational_params(data, approximate_posterior, likelihood, prior, whiten):
    breakpoint()
