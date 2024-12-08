import chex
import jax
import jax.numpy as np
import objax

from ....dispatch import dispatch, evoke, _ensure_str
from .... import settings
from ....utils.utils import fix_block_shapes
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_covar, whitened_gaussian_conditional_diagional, whitened_gaussian_conditional_full, gaussian_conditional_blocks, gaussian_spatial_conditional
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT, batched_block_diagional
from ....data import TemporallyGroupedData, is_timeseries_data

from .meanfield_utils import meanfield_marginal_blocks

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform, Aggregate, Joint
from ....transforms import JointDataLatentPermutation, IndependentDataLatentPermutation, DataLatentPermutation, IndependentJointDataLatentPermutation
from ....transforms.pdes import DifferentialOperatorJoint
from ....transforms.latent_variable import LatentVariable
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian, ConjugatePrecisionGaussian, MeanFieldAcrossDataApproximatePosterior
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity, SpatialSparsity, StackedSparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ...integrals.samples import approximate_expectation
from ....core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part, get_permutated_prior
from ....core.block_types import Block, get_block_type, compare_block_types, get_block_dim
from ...permutations import data_order_to_output_order, permute_mat, permute_vec, permute_vec_blocks

from .linear_marginals import linear_marginal_blocks

from ....data.sequential import add_temporal_points

# ========================= Conjugate Gaussian Approximate Posterior Marginal Blocks =========================

@dispatch(ConjugatePrecisionGaussian, Likelihood, 'GPPrior', Sparsity, whiten=False)
@dispatch(ConjugateGaussian, Likelihood, 'GPPrior', Sparsity, whiten=False)
@dispatch(FullConjugateGaussian, Likelihood, DataLatentPermutation, Sparsity, whiten=False)
@dispatch(FullConjugateGaussian, Likelihood, DataLatentPermutation, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, m, S, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten: bool):
    N = XS.shape[0]

    Q = prior.base_prior.output_dim

    if is_timeseries_data(approximate_posterior.surrogate.data):
        mu, var = approximate_posterior.surrogate.predict_f(XS, diagonal=False, squeeze=False)
        # fix block sizes
        pred_mu, pred_var = fix_block_shapes(mu, var, data, likelihood, approximate_posterior, out_block)

    else:
        # this just predict at the inducing points in time
        XS_temporal_data, sorted_data, _, mu, var = approximate_posterior.surrogate.predict_temporal(XS)
        chex.assert_rank([mu, var], [3, 4])

        # construct data with same temporal points as sort_data but with all the required spatial points

        # construct temporally grouped data so we know what spatial points we need
        xs_data_temp = TemporallyGroupedData(XS, None)

        # we need a dummy spatial point to padd out the temporal predictions as we have to predict across all of time
        dummy_space = xs_data_temp.X_space[0]

        # time x space format
        mu = sorted_data.unsort(mu)[data.Nt:]
        var = sorted_data.unsort(var)[data.Nt:]

        # time-space 
        all_dummy_XS = xs_data_temp.X_st

        # time-space 
        xs_data = TemporallyGroupedData(all_dummy_XS, None, sort=False)

        if _ensure_str(prior) == 'GPPrior':
            prior_parent = Independent([prior])
            sparsity = [sparsity]
            data_x = approximate_posterior.surrogate.data._X 
            approximate_posterior = MeanFieldConjugateGaussian(approximate_posteriors=[approximate_posterior])
        else:
            prior_parent = prior.parent
            data_x =approximate_posterior.surrogate.data._X 

        mu, var = evoke('spatial_conditional', xs_data, prior_parent, prior_parent, approximate_posterior)(
            xs_data, 
            data_x, 
            mu, 
            var[:, 0, ...], 
            approximate_posterior,
            likelihood,
            prior_parent,
            sparsity,
            out_block,
            whiten
        )

        chex.assert_rank([mu, var], [3, 4])


        out_dim = prior.output_dim

        # fix permutations
        mu_p = jax.vmap(lambda a: permute_vec(a, out_dim))(mu)
        var_p = jax.vmap(lambda A: permute_mat(A[0], out_dim))(var)

        mu_p = np.reshape(mu_p, [-1, out_dim, 1])
        var_p = batched_block_diagional(var_p, out_dim)
        var_p = np.reshape(var_p, [-1, 1, out_dim, out_dim])


        # time x space format
        pred_mu = XS_temporal_data.unsort(mu_p)
        pred_var = XS_temporal_data.unsort(var_p)

        chex.assert_equal([pred_mu.shape[0]], [pred_var.shape[0]])
        chex.assert_equal([XS.shape[0]], [pred_mu.shape[0]])

    chex.assert_rank([pred_mu, pred_var], [3, 4])

    return pred_mu, pred_var


# ========================= Gaussian Approximate Posterior Marginal Blocks =========================
@dispatch('DiagonalGaussianApproximatePosterior', Likelihood, 'GPPrior', Sparsity, whiten=False)
@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, m, S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten: bool):
    """ Computes the diagonal of q(f) = ∫ p(f | u) q(u) du """
    chex.assert_rank([m, S_chol], [3, 4])

    S_chol = S_chol[0, 0, ...]
    m = m[:, 0, ...]

    chex.assert_shape([S_chol], [m.shape[0], m.shape[0]])

    if out_block == Block.FULL:
        mu, var = gaussian_conditional(
            XS, 
            data.X, 
            prior.covar(sparsity.Z, sparsity.Z), 
            prior.covar(XS, sparsity.Z), 
            prior.covar(XS, XS), 
            m,
            S_chol,
            prior.mean(sparsity.Z),
            prior.mean(XS),
        )

        # fix shapes
        mu = mu[None, ...]
        var = var[None, None, ...]
    else:


        mu, var = gaussian_conditional_diagional(
            XS, 
            data.X, 
            prior.covar(sparsity.Z, sparsity.Z), 
            prior.covar(XS, sparsity.Z), 
            prior.var(XS), 
            m,
            S_chol,
            prior.mean(sparsity.Z),
            prior.mean(XS),
        )

        # fix shapes
        mu = mu[..., None]
        var = var[..., None, None]

    return mu, var

@dispatch('DiagonalGaussianApproximatePosterior', Likelihood, 'GPPrior', Sparsity, whiten=True)
@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', Sparsity, whiten=True)
def marginal_prediction_blocks(XS, data, m, S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten: bool):
    """ Computes the diagonal of q(f) = ∫ p(f | u) q(u) du """
    chex.assert_rank([m, S_chol], [3, 4])

    S_chol = S_chol[0, 0, ...]
    m = m[:, 0, ...]

    chex.assert_shape([S_chol], [m.shape[0], m.shape[0]])

    #TODO: only works with zero mean gps
    mu, var = whitened_gaussian_conditional_diagional(
        XS, 
        data.X, 
        prior.covar(sparsity.Z, sparsity.Z), 
        prior.covar(XS, sparsity.Z), 
        prior.var(XS)[:, 0], 
        m,
        S_chol
    )

    # fix shapes
    mu = mu[..., None]
    var = var[..., None, None]


    return mu, var

@dispatch(FullGaussianApproximatePosterior, Likelihood, Joint, Sparsity, whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Joint, Sparsity, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DataLatentPermutation, Sparsity, whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DataLatentPermutation, Sparsity, whiten=True)
def marginal_prediction_blocks(XS, data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten: bool):
    """ 
    approximate_posterior is already is in latent data format and so needs to be converted to data-latent format. 
    """
    chex.assert_rank([q_m, q_S], [3, 4])

    q_m = q_m[:, 0, ...]
    q_S = q_S[0, 0, ...]

    prior = get_permutated_prior(prior)
    assert isinstance(prior, DataLatentPermutation)

    if whiten:
        base_prior = prior.base_prior
        Z = base_prior.get_Z_blocks()
        chex.assert_rank(Z, 3)

        Kzz = base_prior.b_covar(Z, Z)
        chex.assert_rank(Kzz, 2)

        Kzz_chol = cholesky(add_jitter(Kzz, settings.jitter))
        chex.assert_equal_shape([Kzz_chol, q_S])

        q_m, q_S =  Kzz_chol @ q_m, Kzz_chol @ q_S


    base_prior = prior.base_prior

    Q = base_prior.output_dim
    M = sparsity[0].shape[0]
    D = XS.shape[-1]
    NS = XS.shape[0]

    # Variational parameters are in latent-data format
    chex.assert_shape(q_m, [M * Q, 1])
    chex.assert_shape(q_S, [M * Q, M * Q])

    # Get all Z in latent-data format
    Z_all = np.array(base_prior.get_Z_stacked())
    Q1, Q2, _, _ = Z_all.shape

    chex.assert_shape(Z_all, [Q1, Q2, M, D])

    # Convert XS to latent_data format
    XS_tiled = np.tile(XS, [Q1, Q2, 1, 1])

    # Z does not need to be ordered, only X
    # Compute non permuted full covariance - this will be block diagonal
    K_zz = prior.np_b_covar(Z_all, Z_all)
    chex.assert_shape(K_zz, [Q*M, Q*M])

    # Compute Kxz with x permutated into data-latent format
    # Left permute x, and do not permute Z
    #TODO: IS THIS CAUSING JIT ISSUES??
    Kxz_p = prior.lp_rb_covar(XS_tiled[0], Z_all)

    chex.assert_shape(Kxz_p, [Q*NS, Q*M])

    # Compute the block diagonals of the permutated Kxx
    # TODO: stop tiling XS here
    K_xx_p = prior.b_full_var_blocks(
        XS_tiled,
        1,
        Q
    )

    chex.assert_shape(K_xx_p, [NS, Q, Q])

    mean_Z = prior.b_mean(Z_all)
    mean_XS = prior.mean(XS)

    # Compute q(F) = \int p(F | U) q(U) dU
    # Comput blocks of
    #val = K_xx - Kxz_p @ cholesky_solve(K_chol, Kxz_p.T)

    # TODO: assuming mean is zero
    _m, _S =  gaussian_conditional_blocks(
        1, 
        Q, 
        XS, 
        Z_all, 
        K_zz, 
        Kxz_p, 
        K_xx_p, 
        q_m,
        q_S,
        mean_Z,
        mean_XS,
    )

    # fix shapes
    _m = _m[..., None]
    _S = _S[:, None, ...]

    return _m, _S


@dispatch(FullConjugateGaussian, Likelihood, Independent, Sparsity, whiten=True)
@dispatch(FullConjugateGaussian, Likelihood, Independent, Sparsity, whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Independent, Sparsity, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Independent, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten: bool):
    # TODO: assuming that sparsity is the same across latents
    sparsity_arr = prior.base_prior.get_sparsity_list()
    base_prior = get_permutated_prior(prior)

    # TODO: refactor to make the structure the same
    if isinstance(base_prior, IndependentJointDataLatentPermutation):
        # we need to a special structure that unpacks the prior 
        fn = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, prior, prior.parent[0], sparsity_arr[0], whiten=whiten)

        # we dont need to pass through the permutated prior as this will be handled down stream
        marginal_mu, marginal_var = fn(
            XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity_arr, out_block, whiten
        ) 

    else:
        fn = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, base_prior, sparsity_arr[0], whiten=whiten)

        marginal_mu, marginal_var = fn(
            XS, data, q_m, q_S_chol, approximate_posterior, likelihood, base_prior, sparsity_arr, out_block, whiten
        ) 
    chex.assert_rank([marginal_mu, marginal_var], [3, 4])

    return marginal_mu, marginal_var


@dispatch(MeanFieldAcrossDataApproximatePosterior, Likelihood, Independent, Sparsity, whiten=True)
@dispatch(MeanFieldAcrossDataApproximatePosterior, Likelihood, Independent, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten: bool):
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    approx_posterior = approximate_posterior.approx_posteriors[0]
    sparsity = StackedSparsity(prior.base_prior.get_sparsity_list()[0])
    base_prior = prior.parent[0]

    fn = evoke('marginal_prediction_blocks', approx_posterior, likelihood, base_prior, sparsity, whiten=whiten)

    # we dont need to pass through the permutated prior as this will be handled down stream
    mu, var = fn(
        XS, data, q_m, q_S_chol, approx_posterior, likelihood, base_prior, sparsity, out_block, whiten
    ) 
    chex.assert_rank([mu, var], [3, 4])

    return mu, var



@dispatch(MeanFieldApproximatePosterior, Likelihood, Independent, Sparsity, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, Independent, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten: bool):
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    marginal_mu, marginal_var = meanfield_marginal_blocks(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block, whiten, 
        XS = XS,
        sparsity=sparsity,
        prediction=True
    )

    chex.assert_rank([marginal_mu, marginal_var], [3, 4])
    return marginal_mu, marginal_var


@dispatch(MeanFieldApproximatePosterior, Likelihood, LinearTransform, Sparsity, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, LinearTransform, Sparsity, whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, LinearTransform, Sparsity, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, LinearTransform, Sparsity, whiten=False)
@dispatch(MeanFieldConjugateGaussian, Likelihood, LinearTransform, Sparsity, whiten=True)
@dispatch(MeanFieldConjugateGaussian, Likelihood, LinearTransform, Sparsity, whiten=False)
@dispatch(FullConjugateGaussian, Likelihood, LinearTransform, Sparsity, whiten=True)
@dispatch(FullConjugateGaussian, Likelihood, LinearTransform, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten: bool):
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    return linear_marginal_blocks(
        data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block, whiten, XS=XS, sparsity=sparsity
    )


@dispatch(MeanFieldApproximatePosterior, Likelihood, Aggregate, Sparsity, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, Aggregate, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity,  out_block: int, whiten: bool):

    def site_fn(XS_group):
        
        prior_parent = prior.parent

        out_block_type = Block.FULL

        mu_p, var_p  = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, prior_parent, sparsity[0], whiten=whiten)(
            XS_group, data, q_m, q_S_chol, approximate_posterior, likelihood, prior_parent, sparsity, out_block_type, whiten
        ) 


        return mu_p, var_p

    marginal_mu, marginal_var = jax.vmap(site_fn, [0])(XS)

    # fix shapes
    marginal_mu = marginal_mu[..., 0]
    marginal_var = marginal_var[:, 0, ...]

    group_size = marginal_mu.shape[2]

    marginal_mu = np.sum(marginal_mu, axis=2)/group_size
    marginal_var = np.sum(np.sum(marginal_var, axis=2), axis=2)/(group_size*group_size)

    marginal_mu = marginal_mu[..., None]
    marginal_var = marginal_var[..., None, None]

    return marginal_mu, marginal_var


@dispatch(ApproximatePosterior, Likelihood, list, Sparsity, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, list, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten: bool):
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    mu_list, var_list = [], []
    for p in prior:
        mu_p, var_p  = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, p, sparsity[0], whiten=whiten, debug=False)(
            XS, data, q_m, q_S_chol, approximate_posterior, likelihood, p, sparsity, out_block, whiten
        ) 

        mu_list.append(mu_p)
        var_list.append(var_p)

    return mu_list, var_list


@dispatch(ApproximatePosterior, Likelihood, Transform, Sparsity, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, Transform, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten: bool):

    # find out if the model is linear or not
    model_type = get_model_type(prior)

    # when non linear we transform up to the last linear transform and then use sampling
    linear_model_part = get_linear_model_part(prior)

    # we are only processing the linear part so we can assume that it is linear
    val = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, linear_model_part, sparsity[0], whiten=whiten)(
        XS, data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, sparsity, out_block, whiten
    ) 

    if isinstance(model_type, LinearModel):
        return val
    
    raise NotImplementedError()

# ========================= Predictions =========================

@dispatch('latents', MeanFieldApproximatePosterior, Likelihood, Transform, whiten=False)
@dispatch('latents', MeanFieldApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch('latents', FullGaussianApproximatePosterior, Likelihood, Transform, whiten=False)
@dispatch('latents', FullGaussianApproximatePosterior, Likelihood, Transform, whiten=True)
def marginal(XS, data, approximate_posterior, likelihood, prior, inference, out_block: Block, whiten: bool):

    sparsity_list = prior.base_prior.get_sparsity_list()
    latents = prior.base_prior

    q_m, q_S_chol = evoke('variational_params', approximate_posterior, likelihood, latents, whiten)(
        data, approximate_posterior, likelihood, latents, whiten
    )
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    mu, var = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, latents, sparsity_list[0], whiten=whiten)(
        XS, data, q_m, q_S_chol, approximate_posterior, likelihood, latents, sparsity_list, out_block , whiten
    )

    return mu, var

@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal_prediction(XS, data, approximate_posterior, likelihood, prior, inference, diagonal, whiten, num_samples=None, posterior=False):

    if num_samples is None:
        num_samples = inference.prediction_samples

    model_type = get_model_type(prior)

    if isinstance(model_type, LinearModel):
        if diagonal == True:
            out_block = Block.DIAGONAL
        else:
            raise NotImplementedError()

        # if the model is linear we can just return here
        sparsity_list = prior.base_prior.get_sparsity_list()

        q_m, q_S_chol = evoke('variational_params', approximate_posterior, likelihood, prior.base_prior, whiten)(
            data, approximate_posterior, likelihood, prior.base_prior, whiten
        )
        chex.assert_rank([q_m, q_S_chol], [3, 4])

        # compute predictions of the part of linear model
        linear_model_part = get_linear_model_part(prior)

        if posterior:
            mu, var = evoke('marginal', approximate_posterior, likelihood,  linear_model_part, whiten=whiten)(
                data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part , whiten
            )
        else:
            mu, var = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, linear_model_part, sparsity_list[0], whiten=whiten)(
                XS, data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, sparsity_list, out_block , whiten
            )

        return mu, var
    else:
        out_block = Block.DIAGONAL

        # set diagonal to True as it doesnt matter is diag=False as we are mean-field

        mu = evoke('marginal_prediction_samples', approximate_posterior, likelihood, prior, whiten=whiten)(
            XS, data, approximate_posterior, likelihood, prior, inference, True, whiten, num_samples = num_samples, posterior=posterior
        )

        out_block_dim = get_block_dim(out_block)

        chex.assert_shape(mu, (num_samples, XS.shape[0], prior.output_dim, out_block_dim))

        if diagonal:
            second_moment =  mu**2

            mu = np.mean(mu, axis=0)
            second_moment = np.mean(second_moment, axis=0)

            mu = np.transpose(mu, [1, 0, 2])
            second_moment = np.transpose(second_moment, [1, 0, 2])

            var = second_moment - np.square(mu)

            # fix shapes back to data-latent format
            mu = np.transpose(mu, [1, 0, 2])
            var = np.transpose(var, [1, 0, 2])

            var = var[..., None]

        else:
            second_moment = mu @ np.transpose(mu, [0, 1, 3, 2])

            mu = np.mean(mu, axis=0)
            second_moment = np.mean(second_moment, axis=0)
            var = second_moment - mu @ np.transpose(mu, [0, 2, 1])

            var = var[:, None, ...]

        chex.assert_rank([mu, var], [3, 4])
        return mu, var

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal_prediction(XS, data, approximate_posterior, likelihood, prior, inference, diagonal, whiten, num_samples=None, posterior=False):


    if num_samples is None:
        num_samples = inference.prediction_samples

    out_block = Block.BLOCK

    model_type = get_model_type(prior)
    if isinstance(model_type, LinearModel):
        sparsity_list = prior.base_prior.get_sparsity_list()

        q_m, q_S_chol = evoke('variational_params', approximate_posterior, likelihood, prior, whiten)(
            data, approximate_posterior, likelihood, prior, whiten
        )
        chex.assert_rank([q_m, q_S_chol], [3, 4])

        # compute predictions of the part of linear model
        linear_model_part = get_linear_model_part(prior)

        if posterior:
            mu, var = evoke('marginal', approximate_posterior, likelihood,  linear_model_part, whiten=whiten)(
                data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part , whiten
            )
        else:
            mu, var = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, linear_model_part, sparsity_list[0], whiten=whiten, debug=False)(
                XS, data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, sparsity_list, out_block , whiten
            )
        if type(linear_model_part) is not list:
            # wrap mu, var in a list 
            mu = [mu]
            var = [var]

        
        mu_list = []
        var_list = []
        for i in range(len(mu)):
            mu_i = mu[i]
            var_i = var[i]
            chex.assert_rank([mu_i, var_i], [3, 4])

            if diagonal:
                var_i = np.transpose(np.diagonal(var_i, axis1=2, axis2=3), [0, 2, 1])[..., None]
            else:
                # no action required as var will already be of the correct shape
                pass

            mu_list.append(mu_i)
            var_list.append(var_i)

        if type(linear_model_part) is not list:
            # unwrap list
            return mu_list[0], var_list[0]

        return mu_list, var_list
    else:
        mu = evoke('marginal_prediction_samples', approximate_posterior, likelihood, prior, whiten=whiten)(
            XS, data, approximate_posterior, likelihood, prior, inference, diagonal, whiten, num_samples = num_samples, posterior=posterior
        )

        mu_samples = np.copy(mu)

        # TODO: fix shapes with aggregation blocks?
        #chex.assert_shape(mu, (num_samples, XS.shape[0], prior.output_dim, 1))

        if diagonal:
            second_moment =  mu**2

            mu = np.mean(mu, axis=0)
            second_moment = np.mean(second_moment, axis=0)

            mu = np.transpose(mu, [1, 0, 2])
            second_moment = np.transpose(second_moment, [1, 0, 2])

            var = second_moment - np.square(mu)

            # fix shapes back to data-latent format
            mu = np.transpose(mu, [1, 0, 2])
            var = np.transpose(var, [1, 0, 2])

            var = var[..., None]

        else:
            second_moment = mu @ np.transpose(mu, [0, 1, 3, 2])

            mu = np.mean(mu, axis=0)
            second_moment = np.mean(second_moment, axis=0)
            var = second_moment - mu @ np.transpose(mu, [0, 2, 1])

            var = var[:, None, ...]

        chex.assert_rank([mu, var], [3, 4])
        return mu, var

# ================================== Samples ==============================

@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal_prediction_samples(XS, data, approximate_posterior, likelihood, prior, inference, diagonal, whiten, num_samples = None, posterior=False):

    if num_samples is None:
        num_samples = inference.prediction_samples 

    if issubclass(type(approximate_posterior.approx_posteriors[0]), FullGaussianApproximatePosterior):
        out_block = Block.BLOCK
    else:
        if diagonal == True:
            out_block = Block.DIAGONAL
        else:
            raise NotImplementedError()

    sparsity_list = prior.base_prior.get_sparsity_list()

    q_m, q_S_chol = evoke('variational_params', approximate_posterior, likelihood, prior.base_prior, whiten)(
        data, approximate_posterior, likelihood, prior.base_prior, whiten
    )
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    # compute predictions of the part of linear model
    linear_model_part = get_linear_model_part(prior)
    model_type = get_model_type(prior)

    if posterior:
        mu, var = evoke('marginal', approximate_posterior, likelihood,  linear_model_part, whiten=whiten)(
            data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part , whiten
        )
    else:
        mu, var = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, linear_model_part, sparsity_list[0], whiten=whiten)(
            XS, data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, sparsity_list, out_block , whiten
        )

    #otherwise we need to sample / use quadrature to compute the remaining integrals
    block_type = compare_block_types(
        get_block_type(1), 
        out_block
    )

    if type(mu) == list:
        # if mu is a list then prior must be a MultiOutput
        mu_res = []
        for i in range(len(mu)):
            mu_i = approximate_expectation(
                lambda f: f, 
                mu[i], 
                var[i], 
                prior = prior.parent[i],
                fn_args = [],
                generator = inference.generator, 
                num_samples = num_samples,
                block_type = block_type,
                average = False
            )
            mu_res.append(mu_i)
        mu = mu_res
        mu = np.transpose(np.stack(mu), [1, 2, 0, 3, 4])[:, :, :, 0, :]
    else:
        mu = approximate_expectation(
            lambda f: f, 
            mu, 
            var, 
            prior = prior,
            fn_args = [],
            generator = inference.generator, 
            num_samples = num_samples,
            block_type = block_type,
            average = False
        )

    return mu

@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, Transform, whiten=False)
def marginal_prediction_samples(XS, data, approximate_posterior, likelihood, prior, inference, diagonal, whiten, num_samples=None, posterior=False):

    if num_samples is None:
        num_samples = inference.prediction_samples 

    out_block = Block.BLOCK

    sparsity_list = prior.base_prior.get_sparsity_list()

    q_m, q_S_chol = evoke('variational_params', approximate_posterior, likelihood, prior, whiten)(
        data, approximate_posterior, likelihood, prior, whiten
    )
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    # compute predictions of the part of linear model
    linear_model_part = get_linear_model_part(prior)
    model_type = get_model_type(prior)

    if posterior:
        mu, var = evoke('marginal', approximate_posterior, likelihood,  linear_model_part, whiten=whiten)(
            data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part , whiten
        )
    else:
        mu, var = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, linear_model_part, sparsity_list[0], whiten=whiten)(
            XS, data, q_m, q_S_chol, approximate_posterior, likelihood, linear_model_part, sparsity_list, out_block , whiten
        )

    #otherwise we need to sample / use quadrature to compute the remaining integrals
    block_type = compare_block_types(
        get_block_type(1), 
        out_block
    )
    
    if type(mu) == list:

        # if mu is a list then prior must be a MultiOutput
        mu_res = []
        for i in range(len(mu)):
            mu_i = approximate_expectation(
                lambda f: f, 
                mu[i], 
                var[i], 
                prior = prior.parent[i],
                fn_args = [],
                generator = inference.generator, 
                num_samples = num_samples,
                block_type = block_type,
                average = False
            )
            mu_res.append(mu_i)
        mu = mu_res
        mu = np.transpose(np.stack(mu), [1, 2, 0, 3, 4])[:, :, :, :, 0]
    else:
        mu = approximate_expectation(
            lambda f: f, 
            mu, 
            var, 
            prior = prior,
            fn_args = [],
            generator = inference.generator, 
            num_samples = num_samples,
            block_type = block_type,
            average = False
        )

    # TODO: fix shapes with aggregation blocks?
    #chex.assert_shape(mu, (num_samples, XS.shape[0], prior.output_dim, 1))

    return mu

# ================================== Marginal Covars ==============================

@dispatch('DiagonalGaussianApproximatePosterior', Likelihood, 'GPPrior', Sparsity, whiten=False)
@dispatch('GaussianApproximatePosterior', Likelihood, 'GPPrior', Sparsity, whiten=False)
def marginal_prediction_covar(X1, X2, data, m, S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    #TODO FIX THIs
    #chex.assert_rank([m, S_chol], [3, 4])

    K12 = prior.kernel.K(X1, X2)
    K1z = prior.kernel.K(X1, sparsity.Z)
    Kz2 = prior.kernel.K(sparsity.Z, X2)
    Kzz = prior.kernel.K(sparsity.Z, sparsity.Z)

    return gaussian_conditional_covar(
        X1, X2, sparsity.Z,
        Kzz, 
        K1z,
        Kz2,
        K12,
        m,
        S_chol[0]
    )

@dispatch(ConjugateGaussian, Likelihood, 'GPPrior', Sparsity, whiten=False)
def marginal_prediction_covar(X1, X2, data, m, S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    # TODO: this is hack, refactor later
    # this is required as we need gradients wrt m, S_chol

    prior = approximate_posterior.surrogate

    lik_var = prior.likelihood.likelihood_arr[0].full_variance

    return gaussian_predictive_covar(Y, K_xs, K_xs_x, K_xx, K_x_xs, mean_x, mean_xs, lik_var)
    #return approximate_posterior.surrogate.covar(X1, X2)

@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, Sparsity, whiten=True)
@dispatch(MeanFieldApproximatePosterior, ProductLikelihood, Independent, Sparsity, whiten=False)
def marginal_prediction_covar(XS_1, XS_2, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    chex.assert_rank([q_m, q_S_chol], [3, 4])
    latents_arr = prior.latents
    approx_posteriors_arr = approximate_posterior.approx_posteriors
    sparsity_arr = sparsity
    likelihood_arr = likelihood.likelihood_arr

    num_latents = len(sparsity_arr)
    N1 = XS_1.shape[0]
    N2 = XS_2.shape[0]

    #TODO: assuming that all likelihoods are the same
    likelihood_arr = [likelihood_arr[0] for q in range(num_latents)]
    whiten_arr = [whiten for q in range(num_latents)]
    out_block_arr = [likelihood_arr[0].block_type for q in range(num_latents)]

    # Compute q(f) for each output
    marginal_var = batch_over_module_types(
        evoke_name = 'marginal_prediction_covar',
        evoke_params = [],
        module_arr = [approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr],
        fn_params = [XS_1, XS_2, data, q_m, q_S_chol, approx_posteriors_arr, likelihood_arr, latents_arr, sparsity_arr, out_block_arr, whiten_arr],
        fn_axes = [None, None, None, 1, 1, 0, 0, 0, 0, 0, 0],
        dim = len(latents_arr),
        out_dim  = 1,
        evoke_kwargs = {'whiten': whiten}
    )

    marginal_var = marginal_var[None, ...]

    chex.assert_shape(marginal_var, [1, prior.output_dim, N1, N2])

    return marginal_var

@dispatch(ApproximatePosterior, Likelihood, LinearTransform, Sparsity, whiten=True)
@dispatch(ApproximatePosterior, Likelihood, LinearTransform, Sparsity, whiten=False)
def marginal_prediction_covar(XS_1, XS_2, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block, whiten):
    chex.assert_rank([q_m, q_S_chol], [3, 4])

    # TODO: assuming indenity transforms

    var_parent  = evoke('marginal_prediction_covar', approximate_posterior, likelihood, prior.base_prior, sparsity[0], whiten=whiten)(
        XS_1, XS_2, data, q_m, q_S_chol, approximate_posterior, likelihood, prior.base_prior, sparsity, out_block, whiten
    ) 

    return var_parent

