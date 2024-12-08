"""
Dispatched functions for computing:
    1) q(u)
    2) q(f) = \int p(f | u) q(u) du

To make computing natural gradients easier we compute the ELL is computed by:
    1) Collecting appropriate paramters from the approximate posterior q(u):
        - (i.e the diagonal, block diagonal, full covariance, etc)
    2) Passing these to the appropiate marginal to compute q(f)
    3) Compute the ELL

This file contains the dispatched method for computing both q(u) and q(f)

The way marginals are computed is very general to support 
    - blocked likelihoods,
    - meanfield and full gaussian approximate posteriors,
    - linear / non linear transformations
    - differential operator transformations

With these a wide variety of variational GP based models can be constructed.
"""
import chex
import jax
import jax.numpy as np
import objax

from ....dispatch import dispatch, evoke
from .... import settings
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_covar, whitened_gaussian_conditional_diagional, whitened_gaussian_conditional_full, gaussian_conditional_blocks, whitened_gaussian_conditional_full
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT, cholesky_solve, triangular_solve, batched_block_diagional, to_block_diag
from ...permutations import left_permute_mat, data_order_to_output_order, permute_vec, permute_mat, unpermute_vec, unpermute_mat
from ....core import Block, get_block_dim

from .meanfield_utils import meanfield_marginal_blocks

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform, Aggregate
from ....transforms.pdes import DifferentialOperatorJoint
from ....transforms import JointDataLatentPermutation, IndependentDataLatentPermutation, DataLatentPermutation, IndependentJointDataLatentPermutation
from ....transforms.latent_variable import LatentVariable
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian, ConjugateApproximatePosterior, FullConjugateGaussian, MeanFieldAcrossDataApproximatePosterior
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity, StackedSparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ....core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part, get_permutated_prior
from ....data import Data

from .linear_marginals import linear_marginal_blocks

# ================================== NoSparsity Entry Points ==============================
@dispatch(ConjugateApproximatePosterior, Likelihood, 'GPPrior', 'NoSparsity', whiten=False)
@dispatch(FullConjugateGaussian, Likelihood, 'GPPrior', 'NoSparsity', whiten=False)
@dispatch(ConjugateApproximatePosterior, Likelihood, Transform, 'NoSparsity', whiten=False)
@dispatch(FullConjugateGaussian, Likelihood, Transform, 'NoSparsity', whiten=False)
def marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """ 
    With conjugate gaussian there is no need to convert from cholesky parameterizations.  
    
    q_m is in time-latent-(space) format. 
    """
    chex.assert_rank([q_m, q_S], [3, 4])

    N = q_m.shape[0]
    Q = prior.output_dim
    block_size = q_S.shape[-1]

    out_block_dim = get_block_dim(
        out_block, 
        approximate_posterior=approximate_posterior,
        likelihood = likelihood
    )
    
    if block_size == 1:
        q_m = np.reshape(q_m, [N, prior.output_dim, 1])
        q_S = np.reshape(q_S, [N, 1, prior.output_dim, prior.output_dim])

        return q_m, q_S

    if out_block_dim in [block_size, Q] :
        # only return the block diagonals across latents 
        # q_m and q_S are in time-latent-space format
        # to convert to data-latent format we first need convert each time point
        # to space-latent format, and then we can just reshape

        if True:
            # convert to time-space-latent
            mu_p = jax.vmap(lambda a: permute_vec(a, Q))(q_m)
            var_p = jax.vmap(lambda A: permute_mat(A[0], Q))(q_S)
        else:
            mu_p = q_m
            var_p = q_S[:, 0, ...]

        if out_block_dim == block_size:
            var_p = var_p[:, None, ...]
            return mu_p, var_p

        # extract block diagonals
        mu_p_bd = np.reshape(mu_p, [-1, Q, 1])
        var_p_bd = batched_block_diagional(var_p, Q)
        var_p_bd = np.reshape(var_p_bd, [-1, 1, Q, Q])

        chex.assert_rank([mu_p_bd, var_p_bd], [3, 4])
        return mu_p_bd, var_p_bd

    raise RuntimeError()

@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', 'NoSparsity', whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """ 
    Catch all for single latent functions with no sparsity and no whitening.

    In general the variational params are stored in cholesky format so we only need to form the full covariance.
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    N = q_m.shape[0]

    out_block_dim = get_block_dim(out_block)

    if out_block_dim == 1:
        # assuming that q_m, q_S corresponds to a (single) full Gaussian
        # therefore the first two dimensions of q_S_chol are just 1
        chex.assert_equal([q_S_chol.shape[0], q_S_chol.shape[0]], [1, 1])
        q_S = diagonal_from_cholesky(q_S_chol[0, 0])

    elif q_S_chol.shape[-1] < out_block_dim:
        raise RuntimeError()

    elif q_S_chol.shape[-1] > out_block_dim:
        # TODO: subsample
        raise RuntimeError()

    # ensure correct shape
    
    q_m = np.reshape(q_m, [N, 1, out_block_dim])
    q_S = np.reshape(q_S, [N, 1, out_block_dim, out_block_dim])

    return q_m, q_S

@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', 'NoSparsity', whiten=True)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """ Catch all for single latent functions with no sparsity but with whitening"""
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    assert len(sparsity) == 1
    sparsity = sparsity[0]

    N = q_m.shape[0]

    q_m = q_m[:, 0, ...]
    q_S_chol = q_S_chol[0, 0, ...]

    chex.assert_rank(q_S_chol, 2)
    chex.assert_shape(q_S_chol, [q_m.shape[0], q_m.shape[0]])

    Kzz = prior.covar(sparsity.Z, sparsity.Z)
    Kzz_chol = cholesky(add_jitter(Kzz, settings.jitter))

    out_block_dim = get_block_dim(out_block)

    if out_block_dim == 1:
        q_m = Kzz_chol @ q_m
        q_S = diagonal_from_cholesky(Kzz_chol @ q_S_chol)
    else:
        raise NotImplementedError()

    # ensure correct shape
    q_m = q_m[..., None]
    q_S = np.reshape(q_S, [N, 1, out_block_dim, out_block_dim])

    return q_m, q_S


@dispatch(FullGaussianApproximatePosterior, Likelihood, Independent, 'DifferentialOperatorJoint', 'NoSparsity', whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, 'IndependentDataLatentPermutation', 'NoSparsity', whiten=False)
# TODO: check this
#@dispatch(FullGaussianApproximatePosterior, Likelihood, 'JointDataLatentPermutation', 'NoSparsity', whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """
    The approximate posterior (and prior) is stored in latent-data format, this is because the prior is generally block diagional.
    However when computing the expected log likelihood, the likelihood decomposes across data points and hence we need in data-latent format.

    q_m, q_S_chol are paramters of a full Gaussian posterior and so is dense:
        q_m: [QN, 1, 1]
        q_S_chol: [1, 1, QN, QN]

    NOTE: this method is only for IndependentDataLatentPermutation, this functions are simply Independent single output GPs
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    # needed as a fix for the DifferentialOperatorJoint case
    prior = get_permutated_prior(prior)

    # [QN , 1]
    q_m = q_m[:, 0, ...]
    # [QN , QN]
    q_S_chol = q_S_chol[0, 0, ...]

    assert isinstance(prior, DataLatentPermutation)

    out_block_dim = get_block_dim(
        out_block, 
        approximate_posterior=approximate_posterior
    )

    m = q_m
    S = q_S_chol @ q_S_chol.T

    Ns = m.shape[0]

    # Use the base prior as the transformation happens in thre ELL for Full posteriors
    num_latents = prior.base_prior.output_dim

    # X is shaped so that all outputs are grouped together
    # We need to instead group by each input

    # [NQ, 1]
    m_p = prior.permute_vec(m, num_latents)
    # [NQ, NQ]
    S_p = prior.permute_mat(S, num_latents)

    # [N, Q]
    m_p = np.reshape(m_p, [-1, num_latents])

    # Extract block diagonals
    # [N, Q, Q]
    S_blocks = get_block_diagonal(S_p, num_latents)

    # [N, Q, 1]
    m_p = m_p[..., None]
    # [N, 1, Q, Q]
    S_blocks = S_blocks[:, None, ...]

    # Assert shapes are correct
    chex.assert_shape(m_p, [Ns/num_latents, num_latents, 1])
    chex.assert_shape(S_blocks, [Ns/num_latents, 1, num_latents, num_latents])

    return m_p, S_blocks

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, 'NoSparsity', whiten=True)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """
    To handle whitened representations with a full gaussian posterior we simple transform the approximate posterior:
       q_m = K^{1/2} m 
       q_S = K^{1/2} S
    and then treat it as an unwhitened posterior.

    NOTE: This means we do not expoloit some of the algebriac simplifications of this form but 
        it means in our implementation we can reuse the marginals already implemented for the unwhitened case.
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    # first reparameterise and then we can treat as usual

    # reparameterise
    base_prior = prior.base_prior

    Z = base_prior.get_Z_blocks()
    chex.assert_rank(Z, 3)

    Kzz = base_prior.b_covar(Z, Z)
    chex.assert_rank(Kzz, 2)

    Kzz_chol = cholesky(add_jitter(Kzz, settings.jitter))

    # fix shapes
    q_m = q_m[:, 0, ...]
    q_S_chol = q_S_chol[0, 0, ...]

    chex.assert_equal_shape([Kzz_chol, q_S_chol])

    # transform 
    q_m, q_S_chol =  Kzz_chol @ q_m, Kzz_chol @ q_S_chol

    # fix back shapes
    q_m = q_m[:, None, ...]
    q_S_chol = q_S_chol[None, None, ...]

    # we have reparemeterised the approximate posterior so we can now treat it as unwhitened
    fn = evoke('marginal_blocks', approximate_posterior, likelihood, prior, sparsity[0], whiten=False)

    return fn(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block, False
    ) 


# ================================== Sparsity Entry Points ==============================
@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', Sparsity, whiten=False)
@dispatch(ApproximatePosterior, Likelihood, 'GPPrior', Sparsity, whiten=True)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """ 
    Catch all for single latent functions with sparsity.

    In general, sparsity can be considered as simply using the predictive distribution for q(f). 
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    M = q_m.shape[0]

    # TODO: block dim

    sparsity = sparsity[0]

    # Call the predictive distribution to compute q(f)
    mu, var =  evoke(
        'marginal_prediction_blocks', approximate_posterior, likelihood, 'GPPrior', sparsity, whiten=False 
    )(
        data.X, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block, whiten
    )

    chex.assert_rank([mu, var], [3, 4])

    return mu, var

#@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, Sparsity, whiten=False)
#@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, Sparsity, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, Sparsity, whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, Sparsity, whiten=True)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """ 
    In general, sparsity can be considered as simply using the predictive distribution for q(f). 
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    M = q_m.shape[0]

    # TODO: block dim

    base_prior = prior.base_prior 

    # Call the predictive distribution to compute q(f)
    mu, var =  evoke(
        'marginal_prediction_blocks', approximate_posterior, likelihood, base_prior, sparsity[0], whiten=False 
    )(
        data.X, data, q_m, q_S_chol, approximate_posterior, likelihood, base_prior, sparsity, out_block, whiten
    )

    chex.assert_rank([mu, var], [3, 4])

    return mu, var


@dispatch(FullConjugateGaussian, Likelihood, Transform, Sparsity, whiten=False)
@dispatch(FullConjugateGaussian, Likelihood, Transform, Sparsity, whiten=True)
def marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """ 
    When using sparsity with a conjugate approximate posterior we handle it in the following way.

    When computing the ELBO we require the marginal q(f), to compute this we use the conjugate property to compute q(u) and then implement the conditionals. This is so we can take gradients through the integral.

    When predicting we can simply use the predictive distribution of the conjugate posterior.
    """
    # TODO: assuming that data_xs and data_x are of the same type
    chex.assert_rank([q_m, q_S], [3, 4])

    #parent is wrapped by a permutator, we don't need this so we pass the parent
    mu, var = evoke('spatial_conditional', data, prior.parent, prior.parent, approximate_posterior)(
        data, 
        sparsity[0].raw_Z, 
        q_m, 
        q_S[:, 0, ...], 
        approximate_posterior,
        likelihood,
        prior.parent,
        sparsity,
        out_block,
        whiten
    )

    #breakpoint()
    out_block_dim = get_block_dim(
        out_block, 
        approximate_posterior=approximate_posterior,
        likelihood = likelihood
    )

    # for testing
    #mu = q_m
    #var = q_S

    Q = prior.base_prior.output_dim
    block_size = var.shape[-1]

    # return either the full var, or the blocks across the latent functions
    if out_block_dim in [Q, block_size]: 
        # convert mu-var to data-latent format and extract block diagonal
        Q = prior.output_dim

        mu_p = jax.vmap(lambda a: permute_vec(a, Q))(mu)
        var_p = jax.vmap(lambda A: permute_mat(A[0], Q))(var)

        if out_block_dim == block_size:
            var_p = var_p[:, None, ...]
            return mu_p, var_p

        mu_p = np.reshape(mu_p, [-1, Q, 1])
        var_p = batched_block_diagional(var_p, Q)
        var_p = np.reshape(var_p, [-1, 1, Q, Q])

        return mu_p, var_p
    else:
        breakpoint()
        raise NotImplementedError()


# ================================== Dispatched q(f) ==============================

@dispatch(FullGaussianApproximatePosterior, Likelihood, Independent, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Independent, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: Block, whiten):
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    # TODO: assuming that sparsity is the same across latents
    sparsity_arr = prior.base_prior.get_sparsity_list()

    base_prior = get_permutated_prior(prior)

    # TODO: refactor to make the structure the same
    if isinstance(base_prior, IndependentJointDataLatentPermutation):
        # we need to a special structure that unpacks the prior 
        fn = evoke('marginal_blocks', approximate_posterior, likelihood, prior, prior.parent[0], sparsity_arr[0], whiten=whiten)

        # we dont need to pass through the permutated prior as this will be handled down stream
        mu, var = fn(
            data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity_arr, out_block, whiten
        ) 

    else:
        fn = evoke('marginal_blocks', approximate_posterior, likelihood, base_prior, sparsity_arr[0], whiten=whiten)

        mu, var = fn(
            data, q_m, q_S_chol, approximate_posterior, likelihood, base_prior, sparsity_arr, out_block, whiten
        ) 
    chex.assert_rank([mu, var], [3, 4])

    return mu, var

@dispatch(GaussianApproximatePosterior, Likelihood, 'GPPrior', whiten=True)
@dispatch(GaussianApproximatePosterior, Likelihood, 'GPPrior', whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: Block, whiten):
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    sparsity_arr = prior.base_prior.get_sparsity_list()

    fn = evoke('marginal_blocks', approximate_posterior, likelihood, prior, sparsity_arr[0], whiten=whiten)

    # we dont need to pass through the permutated prior as this will be handled down stream
    mu, var = fn(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity_arr, out_block, whiten
    ) 
    chex.assert_rank([mu, var], [3, 4])

    return mu, var


@dispatch(MeanFieldAcrossDataApproximatePosterior, Likelihood, Independent, whiten=True)
@dispatch(MeanFieldAcrossDataApproximatePosterior, Likelihood, Independent, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: Block, whiten):
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    approx_posterior = approximate_posterior.approx_posteriors[0]
    sparsity = StackedSparsity(prior.base_prior.get_sparsity_list()[0])
    base_prior = prior.parent[0]

    fn = evoke('marginal_blocks', approx_posterior, likelihood, base_prior, sparsity, whiten=whiten)

    # we dont need to pass through the permutated prior as this will be handled down stream
    mu, var = fn(
        data, q_m, q_S_chol, approx_posterior, likelihood, base_prior, [sparsity], out_block, whiten
    ) 
    chex.assert_rank([mu, var], [3, 4])

    return mu, var

# Mean-field entry point
@dispatch(MeanFieldApproximatePosterior, Likelihood, Independent, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, Independent, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: Block, whiten):
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    marginal_mu, marginal_var = meanfield_marginal_blocks(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block, whiten, prediction=False
    )

    chex.assert_rank([marginal_mu, marginal_var], [3, 4])

    return marginal_mu, marginal_var


# Linear Transform Entry Point
@dispatch(ApproximatePosterior, Likelihood, LinearTransform, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, LinearTransform, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: int, whiten: bool):
    """ 
    Recursively compute the transformed linear marginal.
    This is done by
        1) first done finding the base latent GPs
        2) Computing the full marginal 
        3) transforming this marginal through the remaining linear transforms
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    return linear_marginal_blocks(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block, whiten, XS=None
    )

@dispatch(ApproximatePosterior, Likelihood, Aggregate, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, Aggregate, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: int, whiten: bool):

    def site_fn(X_group, Y_group):
        data_group = Data(X_group, Y_group[None, ...])
        
        prior_parent = prior.parent

        out_block_type = Block.FULL

        mu_p, var_p  = evoke('marginal_blocks', approximate_posterior, likelihood, prior_parent, whiten=whiten)(
            data_group, q_m, q_S_chol, approximate_posterior, likelihood, prior_parent, out_block_type, whiten
        ) 


        return mu_p, var_p

    marginal_mu, marginal_var = jax.vmap(site_fn, [0, 0])(data.X, data.Y)

    # fix shapes
    marginal_mu = marginal_mu[..., 0]
    marginal_var = marginal_var[:, 0, ...]

    group_size = marginal_mu.shape[2]

    marginal_mu = np.sum(marginal_mu, axis=2)/group_size
    marginal_var = np.sum(np.sum(marginal_var, axis=2), axis=2)/(group_size*group_size)

    marginal_mu = marginal_mu[..., None]
    marginal_var = marginal_var[..., None, None]

    return marginal_mu, marginal_var


# list of linear priors
@dispatch(ApproximatePosterior, Likelihood, list, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, list, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: int, whiten: bool):
    """ 
    Recursively compute the transformed linear marginal.  

    Prior is a list of transforms to be comptued Recursively. Simply loop through and collect the results.

    Note we do not use batching and this method will only really be used when the transforms in the list are 
        different, and hence batching wont be applicable anyway.

    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    mu_list, var_list = [], []
    for p in prior:
        mu_p, var_p  = evoke('marginal_blocks', approximate_posterior, likelihood, p, whiten=whiten)(
            data, q_m, q_S_chol, approximate_posterior, likelihood, p, out_block, whiten
        ) 

        mu_list.append(mu_p)
        var_list.append(var_p)

    return mu_list, var_list

# ===============================================================================================
# ===============================================================================================
# ========================================  ENTRY POINTs ========================================
# ===============================================================================================
# ===============================================================================================



# ============================ SINGLE OUTPUT APPROXIMATE POSTERIOR ENTRY POINT ============================

@dispatch(GaussianApproximatePosterior, Likelihood, 'GPPrior', whiten=True)
@dispatch(GaussianApproximatePosterior, Likelihood, 'GPPrior', whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):
    """
    tbd.
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    out_block_type = likelihood.block_type

    sparsity = prior.sparsity

    # we are only processing the linear part so we can assume that it is linear
    mu, var = evoke('marginal_blocks', approximate_posterior, likelihood, prior, sparsity, whiten=whiten)(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block_type, whiten
    ) 

    chex.assert_rank([mu, var], [3, 4])

    return mu, var

# ============================ MEANFIELD APPROXIMATE POSTERIOR ENTRY POINT ============================
@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):
    """
    .
    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    # find out if the model is linear or not
    model_type = get_model_type(prior)

    # when non linear we transform up to the last linear transform and then use sampling
    linear_model_part = get_linear_model_part(prior)

    # get output block type. NonLinear transforms are applied elementwise so only need the likelihood
    #   block size
    out_block: Block = likelihood.block_type

    # we are only processing the linear part so we can assume that it is linear
    val = evoke('marginal_blocks', approximate_posterior, likelihood, linear_model_part, whiten=whiten)(
        data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, out_block, whiten
    ) 

    return val[0], val[1]

# ============================ FULL GAUSSIAN APPROXIMATE POSTERIOR ENTRY POINT ============================
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, whiten: bool):
    """
    When using a Full Gaussian Approximate Posterior the expected log likelihood does not necessarily decompose across data and latents

    There are two situations that we support:
        ProductLikelihoods:
            In this case we assume that the ELL decomposes across datapoints and so  we need to compute the 
                marginals q(f_n) which are of dimension QxQ as they capture the uncertainity across the latents
        BlockDiagonalLikelihood
            In this case the ELL decomposes across the blocks and so we need to compute q(f_n) for each block

    In general we try to push linear transforms into the prior and non-linear into the expected log likelihood. This is because we perform sampling
        in the dispatched ELL code directly.

    """
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    # find out if the model is linear or not
    model_type = get_model_type(prior)

    # when non linear we transform up to the last linear transform and then use sampling
    linear_model_part = get_linear_model_part(prior)

    # Check if the likelihood is a block diagonal or a product likelihood
    if likelihood.block_type == Block.DIAGONAL:
        out_block_type = Block.LATENT
    else:
        out_block_type = likelihood.block_type

    # we are only processing the linear part so we can assume that it is linear
    val = evoke('marginal_blocks', approximate_posterior, likelihood, linear_model_part, whiten=whiten)(
        data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, out_block_type, whiten
    ) 

    return val[0], val[1]



