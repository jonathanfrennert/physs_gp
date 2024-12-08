import chex
import jax
import jax.numpy as np
import objax

from ....utils.utils import fix_block_shapes
from ....dispatch import dispatch, evoke, _ensure_str
from .... import settings
from ....utils.batch_utils import batch_over_module_types
from ...marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_conditional_covar, whitened_gaussian_conditional_diagional, whitened_gaussian_conditional_full, gaussian_conditional_blocks, whitened_gaussian_conditional_full
from ...matrix_ops import diagonal_from_cholesky, get_block_diagonal, block_diagonal_from_cholesky, block_from_vec, cholesky, add_jitter, diagonal_from_XDXT, cholesky_solve, triangular_solve, batched_block_diagional
from ...permutations import left_permute_mat, data_order_to_output_order, permute_vec, permute_mat, unpermute_vec, unpermute_mat, right_permute_mat, permute_mat_ld_to_dl, permute_vec_ld_to_dl
from ....core import Block, get_block_dim
from ...kernel_ops import _batched_st_kernel, _batched_diff_kernel

# Import Types
from ....transforms import Transform, LinearTransform, Independent, NonLinearTransform, Aggregate
from ....transforms.pdes import DifferentialOperatorJoint
from ....transforms import JointDataLatentPermutation, IndependentDataLatentPermutation, DataLatentPermutation
from ....transforms.latent_variable import LatentVariable
from ....approximate_posteriors import ApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, MeanFieldConjugateGaussian, ConjugateApproximatePosterior, FullConjugateGaussian
from ....likelihood import Likelihood, ProductLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood
from ....sparsity import FreeSparsity, Sparsity
from ...integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ....core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part, get_permutated_prior
from ....sparsity import SpatialSparsity, NoSparsity
from ....data import SpatioTemporalData

from .linear_marginals import linear_marginal_blocks

# =============================================================================
# ===================================CVI=======================================
# =============================================================================

# ===========================TRAINING MARGINALS================================

@dispatch(FullConjugateGaussian, Likelihood, Independent, DifferentialOperatorJoint, SpatialSparsity, whiten=False)
def marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):

    if out_block == Block.LATENT or out_block == Block.DIAGONAL:
        out_block_dim = 1
        mu, var = evoke('spatial_conditional', data, prior, prior.parent[0], approximate_posterior)(
            data, 
            sparsity[0].raw_Z, 
            q_m, 
            q_S[:, 0, ...], 
            approximate_posterior,
            likelihood,
            prior,
            sparsity,
            out_block_dim,
            whiten
        )

        chex.assert_rank([mu, var], [3, 4])

        return mu, var
    else:
        # This path is used when, for example, computing the ELL of the surrogate likelihood
        P = q_m.shape[1]


        Ns =  data.Ns
        dummy_prior = prior.parent[0]
        Q = len(prior.parent)

        if dummy_prior.hierarchical:
            # when hierarchical the prior is only defined on time so we only need the time dimension
            space_prior = dummy_prior
            time_prior = dummy_prior.parent
            ds = 1
            dt = time_prior.output_dim
        else:
            # when not hierarchical the prior is only defined on space and time
            space_prior = dummy_prior
            time_prior = dummy_prior.parent
            ds = space_prior.output_dim
            dt = time_prior.output_dim


        # posterior is in [time - Q - dt - ds - space ] format
        # we need it in [time - space - Q - dt - ds] format

        mu_p = jax.vmap(lambda a: permute_vec_ld_to_dl(a, num_latents=Q * dt * ds, num_data=Ns))(q_m)
        var_p = jax.vmap(lambda A: permute_mat_ld_to_dl(A[0], num_latents=Q * dt * ds, num_data=Ns))(q_S)

        var_p = var_p[:, None, ...]

        return mu_p, var_p

@dispatch(FullConjugateGaussian, Likelihood, Independent, DifferentialOperatorJoint, NoSparsity, whiten=False)
def marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block: Block, whiten):
    """
    Args:
        q_m: [Nt, Q - Dt - Ns - Ds, 1]
        q_S: [Nt, 1, Q - Dt - Ns - Ds, Q - Dt - Ns - Ds]
    """

    #dummy base prior
    base_prior = prior.parent[0].base_prior
    hierachial_base_prior = prior.parent[0].hierarchical_base_prior

    Q = len(prior.parent)
    L = base_prior.output_dim
    QL = Q*L

    if settings.whiten_space:
        Kzz = hierachial_base_prior.derivative_kernel.K(data.X_space, data.X_space)
        Kzz_chol = cholesky(add_jitter(Kzz, settings.jitter))

    if out_block == Block.DIAGONAL:
        out_block = Block.LATENT


    # TODO: figure out the format of q_m and q_S
    if (out_block == Block.FULL or out_block == Block.BLOCK):
        mu_p = jax.vmap(lambda a: permute_vec(a, QL))(q_m)
        var_p = jax.vmap(lambda A: permute_mat(A[0], QL))(q_S)

        var_p = var_p[:, None, ...]
        return mu_p, var_p
    elif (out_block == Block.LATENT):
        if not hierachial_base_prior.hierarchical:
            chex.assert_rank([q_m, q_S], [3, 4])
            chex.assert_equal(q_S.shape[1], 1)

            # q_m is in time - latent - space format
            N = data.N
            Nt, _, _= q_m.shape
            Q = prior.output_dim
            _, _, QNs, _ = q_S.shape

            # ensure correct dim
            q_m = np.reshape(q_m, [Nt, QNs, 1])

            # convert to time-space-latent format
            mu_p = jax.vmap(lambda a: permute_vec(a, Q))(q_m)
            var_p = jax.vmap(lambda A: permute_mat(A[0], Q))(q_S)

            # extract block diagonals
            mu_p_bd = np.reshape(mu_p, [-1, Q, 1])
            var_p_bd = batched_block_diagional(var_p, Q)
            var_p_bd = np.reshape(var_p_bd, [-1, 1, Q, Q])


            chex.assert_rank([mu_p_bd, var_p_bd], [3, 4])
            return mu_p_bd, var_p_bd
        else:
            out_block_dim = 1
            mu, var = evoke('spatial_conditional', data, prior, prior.parent[0], approximate_posterior)(
                data, 
                sparsity[0].raw_Z, 
                q_m, 
                q_S[:, 0, ...], 
                approximate_posterior,
                likelihood,
                prior,
                sparsity,
                out_block_dim,
                whiten
            )

            chex.assert_rank([mu, var], [3, 4])

            return mu, var

    raise NotImplementedError()

# DifferentialOperatorJoint with CVI approximate posteriors
@dispatch(FullConjugateGaussian, Likelihood, DifferentialOperatorJoint, whiten=True)
@dispatch(FullConjugateGaussian, Likelihood, DifferentialOperatorJoint, whiten=False)
def marginal_blocks(data, q_m, q_S, approximate_posterior, likelihood, prior, out_block: Block, whiten: bool):
    """ Single Latent CVI model """
    sparsity =  prior.base_prior.get_sparsity_list()

    # Wrap prior in an Independent transform to bring into the same structure as a multi-output ones

    ind_prior = Independent([prior])

    # handle the CVI marginal based on sparsity
    mu, var =  evoke('marginal_blocks', approximate_posterior, likelihood, ind_prior, prior, sparsity[0], whiten=whiten, debug=False)(
        data, q_m, q_S, approximate_posterior, likelihood, ind_prior, sparsity, out_block, whiten
    )

    chex.assert_rank([mu, var], [3, 4])

    return mu, var

# ===========================PREDICTION MARGINALS================================

@dispatch(FullConjugateGaussian, Likelihood, Independent, DifferentialOperatorJoint, Sparsity, whiten=True)
@dispatch(FullConjugateGaussian, Likelihood, Independent, DifferentialOperatorJoint, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block: int, whiten: bool):
    dummy_prior = prior.parent[0]

    if dummy_prior.hierarchical:
    #if True:
        # when hierarchical we need to compute the conditional with the spatial derivate kernels
        # to do this we first predict in time and then compute the required conditional

        # predict in time first
        xs_spatial_data, all_temporal_data, _, pred_mu_train_test, pred_var_train_test = approximate_posterior.surrogate.predict_temporal(XS)
        chex.assert_rank([pred_mu_train_test, pred_var_train_test], [3, 4])


        # from now on we treat and space and time independently
        # so first unpack the temporal predictions and unsort back onto the testing locations
        # then compute the spatial conditions

        # remove the training data
        pred_mu = all_temporal_data.unsort(pred_mu_train_test)[data.Nt:]
        pred_var = all_temporal_data.unsort(pred_var_train_test)[data.Nt:]

        out_block_dim = 1
        mu, var = evoke('spatial_conditional', xs_spatial_data, prior, dummy_prior, approximate_posterior)(
            xs_spatial_data, 
            sparsity[0].raw_Z, 
            pred_mu, 
            pred_var[:, 0, ...], 
            approximate_posterior,
            likelihood,
            prior,
            sparsity,
            out_block_dim,
            whiten
        )

        #mu, var are in [time-space-Q-latent] format
        # unsort to original permutation in XS
        # mu_p_unsorted, mu_p_unsorted are in [data-Q-latent] format
        mu_p_unsorted = xs_spatial_data.unsort(mu)
        var_p_unsorted = xs_spatial_data.unsort(var)

        return mu_p_unsorted, var_p_unsorted

    else:
        pred_mu, pred_var = approximate_posterior.surrogate.predict_f(XS, diagonal=False, squeeze=False)
        chex.assert_rank([pred_mu, pred_var], [3, 4])

        pred_mu, pred_var = fix_block_shapes(pred_mu, pred_var, data, likelihood, approximate_posterior, out_block)
        chex.assert_rank([pred_mu, pred_var], [3, 4])


        return pred_mu, pred_var


@dispatch(FullConjugateGaussian, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=True)
@dispatch(FullConjugateGaussian, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S, approximate_posterior, likelihood, prior, sparsity, out_block: int, whiten: bool):

    # Wrap prior in an Independent transform to bring into the same structure as a multi-output ones

    ind_prior = Independent([prior])

    # TODO: why not already a list?
    if type(sparsity) is not list:
        sparsity = [sparsity]

    # handle the CVI marginal based on sparsity
    mu, var =  evoke('marginal_prediction_blocks', approximate_posterior, likelihood, ind_prior, prior, sparsity[0], whiten=whiten)(
        XS, data, q_m, q_S, approximate_posterior, likelihood, ind_prior, sparsity, out_block, whiten
    )

    chex.assert_rank([mu, var], [3, 4])

    return mu, var


    if True:
        # predict in time first
        mu, var = approximate_posterior.surrogate.predict_f(XS, diagonal=False, squeeze=False, sort_output=False)
        chex.assert_rank([mu, var], [3, 4])

        # fix block sizes
        pred_mu, pred_var = fix_block_shapes(mu, var, data, likelihood, approximate_posterior, out_block)
        chex.assert_rank([pred_mu, pred_var], [3, 4])

        q_m, q_S = pred_mu, pred_var


    if not prior.hierarchical:
        # When not hierarchical predict_f will already compute all teh required derivates
        chex.assert_rank([q_m, q_S], [3, 4])
        chex.assert_equal(q_S.shape[1], 1)

        # q_m is in time - latent - space format
        N = data.N
        Nt, _, _= q_m.shape
        Q = prior.output_dim

        # convert to time-space-latent format
        mu_p = jax.vmap(lambda a: permute_vec(a, Q))(q_m)
        var_p = jax.vmap(lambda A: permute_mat(A[0], Q))(q_S)

        # extract block diagonals
        mu_p_bd = np.reshape(mu_p, [-1, Q, 1])
        var_p_bd = batched_block_diagional(var_p, Q)
        var_p_bd = np.reshape(var_p_bd, [-1, 1, Q, Q])

        chex.assert_rank([mu_p_bd, var_p_bd], [3, 4])
        return mu_p_bd, var_p_bd

    else:
        data_xs = SpatioTemporalData(X=XS, Y=None, sort=True)
        if _ensure_str(data) == 'Data':
            breakpoint()
            # TODO: why is this necessary? :(
            q_m = np.transpose(q_m, [0, 2, 1])
            return q_m, q_S

        # compute spatial conditonal
        sparsity =  prior.base_prior.get_sparsity_list()

        out_block_dim = 1
        mu, var = evoke('spatial_conditional', data, prior, prior, prior, approximate_posterior)(
            data_xs, 
            sparsity[0].raw_Z, 
            q_m, 
            q_S[:, 0, ...], 
            approximate_posterior,
            likelihood,
            prior,
            sparsity,
            out_block_dim,
            whiten
        )

        breakpoint()

        chex.assert_rank([mu, var], [3, 4])
        return mu, var



# =============================================================================
# ===============================NON-CVI=======================================
# =============================================================================
# DifferentialOperatorJoint with (non-CVI) approximate posteriors
@dispatch(MeanFieldApproximatePosterior, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DifferentialOperatorJoint, Sparsity, whiten=False)
def marginal_prediction_blocks(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity, out_block: int, whiten: bool):
    """
    When we are not in the CVI setting we do not make any distinction between space and time derivates.

    We we are not hierarchical then q(U) is defined over all of the derivavates and can reuse all the marginals from the multi-output setting.

    If we are hierarchical then we compute q(u) and construct q(F) as

        q(F) = \int p(F | u) q(u) \du

    where

        p(F | u) = N( F | K^∇ K^∇_Z u, .)

    which is the same form as the standard required marginals just with a `multi-output' kernel.
    """
    if not prior.hierarchical:
        sparsity_arr = prior.base_prior.get_sparsity_list()
        sparsity_type = sparsity_arr[0]

        base_prior = get_permutated_prior(prior)

        fn = evoke('marginal_prediction_blocks', approximate_posterior, likelihood, base_prior, sparsity_type, whiten=whiten, debug=False)

        mu, var = fn(XS, data, q_m, q_S_chol, approximate_posterior, likelihood, base_prior, sparsity_arr, out_block, whiten)

        chex.assert_rank([mu, var], [3, 4])
        return mu, var

    else:
        chex.assert_rank([q_m, q_S_chol], [3, 4])

        sparsity_arr = prior.base_prior.get_sparsity_list()
        sparsity_type = sparsity_arr[0]

        Z = sparsity_arr[0].Z
        Q = prior.derivative_kernel.output_dim

        NS = XS.shape[0]
        M = Z.shape[0]

        # compute multi-output kernels

        # latent-data format
        # due to a jit bug we have to use the full gaussian_conditional
        #   but we only care about the diagonal of the result so we just compute
        #   the diagonal variance here

        # [Dt x Ds x N] x [Dt x Ds x N]
        Kxx = prior.covar(XS, XS) 
        mean_xx = prior.mean(XS)

        # Dt
        base_prior_output = prior.base_prior.output_dim
        # Ds
        prior_added_output = prior.derivative_kernel.d_computed

        # [Dt x M] x [Dt x M]
        Kzz = prior.base_prior.covar(Z, Z)
        mean_zz = prior.base_prior.mean(Z)

        # [Dt x Dt x N] x [Dt x Ds x M]
        Kxz = prior.covar(XS, Z)
        M = Z.shape[0]

        # remove added outputs to Kxz Z dimension as Kzz is only defined on the base prior

        # [Dt - Ds - N] x [Dt] x [Ds] x [M] -> [Ds - Dt - N] x [ Dt - M]
        Kxz = np.reshape(Kxz, [Kxz.shape[0], base_prior_output, prior_added_output, M])[:, :, 0, :].reshape([Kxz.shape[0], q_m.shape[0]])

        # compute standard marginals
        if whiten:
            mu, var = whitened_gaussian_conditional_full(
                XS, 
                Z, 
                Kzz, 
                Kxz, 
                Kxx, 
                q_m[..., 0],
                q_S_chol[0, 0, ...]
            )
            # TODO: why the transpose?
            P = data_order_to_output_order(NS, prior.output_dim)
            post_mu = np.reshape(P.T @ mu, [NS, prior.output_dim, 1])

            post_var = P.T @ var @ P
            post_var = get_block_diagonal(post_var, prior.output_dim)
            post_var = np.reshape(post_var, [NS, 1, prior.output_dim, prior.output_dim])

            return post_mu, post_var
        else:
            # TODO: this is inefficiently implemented but it works 
            # TODO: why not just vmap over x? this will automatically get the correct format, and may even make the graph smaller?
            P = data_order_to_output_order(NS, prior.output_dim)

            # [N] x [Dt x Dt] x [Dt x Ds]
            Kxx_diag = jax.vmap(lambda x: prior.covar(x, x))(XS[:, None, ...])

            # N x [ Dt x Dt] 
            mu, var = gaussian_conditional_blocks(
                1.0,
                prior.output_dim,
                XS, 
                Z, 
                Kzz, # [Dt x M] x [Dt x M]
                P.T @ Kxz, # [N x  Dt x Ds] x [Dt x M]
                Kxx_diag, # N x [Dt x Ds] x [Ds x Dt]
                q_m[..., 0], # [Dt x M] x 1
                q_S_chol[0, 0, ...], # [Dt x M] x [Dt x M]
                mean_zz, 
                mean_xx
            )


            post_mu = mu[..., None]
            post_var = var[:, None, ...]
            return post_mu, post_var



# DifferentialOperatorJoint with (non-CVI) approximate posteriors
@dispatch(MeanFieldApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=True)
@dispatch(MeanFieldApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=False)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=True)
@dispatch(FullGaussianApproximatePosterior, Likelihood, DifferentialOperatorJoint, whiten=False)
def marginal_blocks(data, q_m, q_S_chol, approximate_posterior, likelihood, prior, out_block: int, whiten: bool):
    """
    DifferentialOperatorJoint with (non-CVI) approximate posteriors.

    If we are hierarchical then we need to compute the required derivatives so we call the necessary diff-op predictions.

    If we are not hierarchical,  q(F) is defined over all the required derivatives, and we are in the standard multi-output setting
        hence we just call the standard marginals
    """
    if prior.hierarchical:
        sparsity_arr = prior.base_prior.get_sparsity_list()
        sparsity_type = sparsity_arr[0]

        mu, var =  evoke('marginal_prediction_blocks', approximate_posterior, likelihood, prior, sparsity_type, whiten=whiten)(
            data.X, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity_arr, out_block, whiten
        )

        chex.assert_rank([mu, var], [3, 4])
        return mu, var
    else:
        # Call standard multi-output marginals
        sparsity_arr = prior.base_prior.get_sparsity_list()
        sparsity_type = sparsity_arr[0]

        base_prior = get_permutated_prior(prior)

        fn = evoke('marginal_blocks', approximate_posterior, likelihood, base_prior, sparsity_type, whiten=whiten, debug=False)

        mu, var = fn(data, q_m, q_S_chol, approximate_posterior, likelihood, base_prior, sparsity_arr, out_block, whiten)

        chex.assert_rank([mu, var], [3, 4])
        return mu, var


 
