""" Kronecker structured conditionals """
from ..dispatch import dispatch, evoke, _ensure_str
from ..utils.batch_utils import batch_over_module_types
from ..utils.utils import get_batch_type
from .marginals import gaussian_conditional_diagional, gaussian_conditional, gaussian_spatial_conditional_diagional, gaussian_spatial_conditional, gaussian_linear_operator_spatial_conditional, gaussian_conditional_blocks, gaussian_linear_operator_spatial_conditional_blocks, gaussian_spatial_conditional_cholesky, gaussian_spatial_conditional_inv, gaussian_linear_operator_spatial_conditional_no_S_cholesky, gaussian_linear_operator_spatial_conditional_blocks_avoid_S_chol
from .matrix_ops import batched_block_diagional, to_block_diag, add_jitter, cholesky, get_block, mat_inv
from .permutations import permute_vec, permute_mat, data_order_to_output_order
from .. import settings 

# Import Types
from ..data import Data, Input, TemporallyGroupedData
from ..approximate_posteriors import MeanFieldApproximatePosterior, FullGaussianApproximatePosterior,FullConjugateGaussian, ConjugateGaussian
from ..likelihood import Likelihood
from ..models import BatchGP, BASE_SDE_GP
from ..transforms import Independent, Joint
from ..transforms.pdes import DifferentialOperatorJoint, PDE
from ..transforms.sdes import SDE
from .kernel_ops import _batched_diff_kernel, _batched_st_kernel

import jax
import jax.numpy as np
from jax import jit, vmap, lax
import chex
import objax
from batchjax import batch_or_loop, BatchType




def spatial_conditional_block(data_xs, data_x, pred_mean, pred_var, prior, batch_space = False):
    """
    Let P be the number of outputs then:

    In:
        pred_mean: Nt x P*Ns x 1
        pred_var: Nt x P*Ns x P*P

    where pred_mean, pred_var are in time-latent-space format.    


    Computes:
        mu = [ I ⊗ Ksz Kzz⁻¹ ] m
        var =  Ktt  ⊗ diag[ Kss - Ksz Kzz⁻¹ Kss] - diag[ Ksz Kzz⁻¹ Stt Kzz⁻¹ Kzs ]^T_t

    """
    # TODO: assuming that prior is independent

    XS_time = data_xs.X_time

    # Get spatial locations with dummy time dimension so kernel evaluations are correct
    XS_space = data_xs.X_space
    X_space = data_x.X_space

    if batch_space:
        # Nt x Ns x D
        XS_space = np.concatenate([np.zeros_like(XS_space[..., [0]]), XS_space], axis=2)
        # Ns x D
        X_space = np.hstack([ np.zeros_like(X_space[..., [0]]), X_space ])
    else:
        # Ns x D
        XS_space = np.hstack([np.zeros([XS_space.shape[0], 1]), XS_space])
        # Ns x D
        X_space = np.hstack([np.zeros([X_space.shape[0], 1]), X_space])

    Ns = X_space.shape[0]

    if batch_space:
        Nss = XS_space.shape[1]
    else:
        Nss = XS_space.shape[0]

    # Precompute all kernels

    # TODO: how to compute the diagonal diff op kernels effeciently

    if batch_space:
        Kss = jax.vmap(lambda xs: _batched_st_kernel(xs, xs, prior, 'spatial', full=True))(XS_space)
        Kss = np.transpose(Kss, [1, 0, 2, 3])
        # latent - space format
        Ksz = jax.vmap(lambda xs: _batched_st_kernel(xs, X_space, prior, 'spatial', full=True))(XS_space)
        Ksz = np.transpose(Ksz, [1, 0, 2, 3])
    else:
        # latent - space format
        Kss = _batched_st_kernel(XS_space, XS_space, prior, 'spatial', full=True)
        # latent - space format
        Ksz = _batched_st_kernel(XS_space, X_space, prior, 'spatial', full=True)

    # latent - time format
    Ktt = _batched_st_kernel(XS_time, XS_time, prior, 'temporal', full=False)
    # latent - space format
    Kzz = _batched_st_kernel(X_space, X_space, prior, 'spatial', full=True)


    # in latent-space format
    if batch_space:
        Kss_full = jax.vmap(to_block_diag)(np.transpose(Kss, [1, 0, 2, 3]))
        Ksz_full = jax.vmap(to_block_diag)(np.transpose(Ksz, [1, 0, 2, 3]))
    else:
        Kss_full = to_block_diag(Kss)
        Ksz_full = to_block_diag(Ksz)

    Kzz_full = to_block_diag(Kzz)


    Q = Kzz.shape[0]

    # if the temporal kernel is a derivate kernel this will return a rank 3 matrix
    if len(Ktt.shape) == 2:
        f_only_flag: bool = True
    else:
        f_only_flag: bool = False

    if f_only_flag:
        # time - latent format
        Ktt = Ktt.T
        # time - latent - space format
        Ktt_full = jax.vmap(
            lambda _ktt: to_block_diag(jax.vmap(
                lambda _k: _k*np.ones([Nss, Nss]),
                0
            )(_ktt)),
            0
        )(Ktt)
    else:
        Ktt_full = Ktt[0]

    if False:
        if not f_only_flag:
            Kzz_full = Kzz_full[:Ns, ...][..., :Ns]
            Ksz_full = Ksz_full[..., :Ns]


    # TODO: check this
    mean_x = np.zeros([pred_mean.shape[1], 1])

    # prior.temporal_output_dim actually returns the state dim across all latents
    if batch_space:
        mean_xs = np.zeros([int((prior.temporal_output_dim/Q)) * Kss_full.shape[1], 1])
    else:
        mean_xs = np.zeros([int((prior.temporal_output_dim/Q)) * Kss_full.shape[0], 1])

    # compute cholesky at each time stamp
    pred_var_chol = jax.vmap(
        lambda S: cholesky(add_jitter(S, settings.jitter)),
        0,
    )(pred_var)

    # batch over time

    if f_only_flag:
        #spatial_fn = gaussian_spatial_conditional
        if True:
            spatial_fn = gaussian_spatial_conditional_cholesky
            Kzz_mat = cholesky(add_jitter(Kzz_full, settings.jitter))
        else:
            spatial_fn = gaussian_spatial_conditional_inv
            Kzz_mat = mat_inv(Kzz_full)

        S = pred_var_chol
    else:
        if settings.avoid_s_cholesky:
            spatial_fn = gaussian_linear_operator_spatial_conditional_no_S_cholesky
            S = pred_var
        else:
            spatial_fn = gaussian_linear_operator_spatial_conditional
            S = pred_var_chol

        Kzz_mat = Kzz_full

    # TODO: derive proper mean 

    if False:
        if np.any(np.array(prior.whiten_space)):
            # whiten transform in space
            Kzz_chol = cholesky(add_jitter(Kzz_full, settings.jitter))
            # TODO: fix the hardcoded time dim
            Kzz_chol = np.kron(np.eye(2), Kzz_chol)
            pred_mean, pred_var_chol =  Kzz_chol @ pred_mean, Kzz_chol @ pred_var_chol

    if batch_space:
        #batch over XS_space, Ksz_full, Kss_full, Ktt_full, pred_mean, pred_var_chol
        batch_arr = [0, None, None, 0, 0, 0, 0, 0, None, None]
    else:
        #batch over Ktt_full, pred_mean, pred_var_chol
        batch_arr = [None, None, None, None, None, 0, 0, 0, None, None]

    mu, var = jax.vmap(
        spatial_fn,
        batch_arr,
    )( 
        XS_space, 
        X_space, 
        Kzz_mat, 
        Ksz_full, 
        Kss_full, 
        Ktt_full, #batching 
        pred_mean, #batching
        S, #batching -- either pred_var or pred_var_chol
        mean_x, 
        mean_xs
    )

    # in time-latent-space format
    var = var[:, None, ...]

    chex.assert_rank([mu, var], [3, 4])
    return mu, var


@dispatch(Data, Input, BASE_SDE_GP, 'GPPrior')
@dispatch(Data, Data, BASE_SDE_GP, 'GPPrior')
@dispatch(Data, Data, BASE_SDE_GP, SDE)
@dispatch(Data, Data, BASE_SDE_GP, PDE)
def spatial_conditional(data_xs: 'Data', data_x: 'Data', pred_mean, pred_var, gp, diagonal):
    """
    gp is a GP prior with a spatio-temporal kernel 

    Computes:
        mu = [ I ⊗ Ksz Kzz⁻¹ ] m
        var = diag[ Ktt ] ⊗ diag[ Kss - Ksz Kzz⁻¹ Kss] - diag[ Ksz Kzz⁻¹ Stt Kzz⁻¹ Kzs ]^T_t
    """

    if _ensure_str(data_xs) == 'TemporallyGroupedData':
        batch_space=True
    else:
        batch_space = False

    mu, var = spatial_conditional_block(data_xs, data_x, pred_mean, pred_var, gp.prior, batch_space=batch_space)
    return mu, var

@dispatch(Input, Independent, Independent, FullGaussianApproximatePosterior)
@dispatch(Data, Independent, Independent, FullGaussianApproximatePosterior)
def spatial_conditional(
    data_xs, 
    data_x, 
    pred_mean, 
    pred_var, 
    approximate_posterior, 
    likelihood, 
    prior, 
    sparsity,
    out_block_dim, 
    whiten
):
    """
    Let P be the number of outputs then:

    In:
        pred_mean: Nt x Ns*P x 1
        pred_var: Nt x Ns*P x Ns*P

    where pred_mean, pred_var are in latent-data format.    
    """

    mu, var = spatial_conditional_block(data_xs, data_x, pred_mean, pred_var, prior)
    chex.assert_rank([mu, var], [3, 4])
    return mu, var

@dispatch(TemporallyGroupedData, Independent, Independent, MeanFieldApproximatePosterior)
@dispatch(TemporallyGroupedData, Independent, Independent, FullGaussianApproximatePosterior)
def spatial_conditional(
    data_xs, 
    data_x, 
    pred_mean, 
    pred_var, 
    approximate_posterior, 
    likelihood, 
    prior, 
    sparsity,
    out_block_dim, 
    whiten
):
    """
    Let P be the number of outputs then:

    In:
        pred_mean: Nt x Ns*P x 1
        pred_var: Nt x Ns*P x Ns*P

    where pred_mean, pred_var are in latent-data format.    
    """

    mu, var = spatial_conditional_block(data_xs, data_x, pred_mean, pred_var, prior, batch_space=True)
    chex.assert_rank([mu, var], [3, 4])
    return mu, var

def differential_spatial_conditional(
    data_xs, 
    data_x, 
    pred_mean, 
    pred_var, 
    approximate_posterior, 
    likelihood, 
    prior, 
    sparsity,
    out_block_dim, 
    whiten,
    batch_space = False
):
    """
    Posterior is in [time - Q - df - ds - space ] format
        we need it in [time - space - Q - df - ds] format

    NOTE: assuming that data_xs and data_x lie at the same temporal locations
    """
    XS_time = data_xs.X_time

    # Data is the sparsity object
    #X_space = sparsity[0].raw_Z.X_space
    # Get spatial locations with dummy time dimension so kernel evaluations are correct
    XS_space = data_xs.X_space
    X_space = data_x.X_space
    X_time = data_xs.X_time

    space_dim = X_space.shape[1]
    Ns = data_x.Ns

    if batch_space:
        Nss = XS_space.shape[1]
    else:
        Nss = XS_space.shape[0]

    if batch_space:
        # Nt x Ns x D
        XS_space = np.concatenate([np.zeros_like(XS_space[..., [0]]), XS_space], axis=2)
        # Ns x D
        X_space = np.hstack([ np.zeros_like(X_space[..., [0]]), X_space ])
    else:
        # Ns x D
        XS_space = np.hstack([np.zeros([XS_space.shape[0], 1]), XS_space])
        # Ns x D
        X_space = np.hstack([np.zeros([X_space.shape[0], 1]), X_space])

    X_time = np.hstack([X_time[:, None], np.zeros([X_time.shape[0], 1])])

    # TODO: assuming that data_xs and data_x are the same

    # TODO: enforce that dummy prior must be a 
    # DifferentialOperatorJoint[DifferentialOperatorJoint[]]

    dummy_prior = prior.parent[0]
    hierarchical = dummy_prior.hierarchical

    # compute the outputs of each of the components of prior
    Q = len(prior.parent)
    base_prior_output = dummy_prior.parent.output_dim
    prior_added_output = dummy_prior.derivative_kernel.d_computed
    out_dim = base_prior_output * prior_added_output

    K_x_t, K_spatial_ss, K_spatial_sz, K_base_spatial_zz = _batched_diff_kernel(prior, X_time, XS_space, X_space, hierarchical, batch_space=batch_space)

    # compute cholesky at each time stamp
    pred_var_chol = jax.vmap(
        lambda S: cholesky(add_jitter(S, settings.jitter)),
        0,
    )(pred_var)

    # TODO: check this
    mean_x = np.zeros([pred_mean.shape[1], 1])
    mean_xs = np.zeros([Nss * out_dim*Q, 1])

    if batch_space:
        batch_arr = [None, None, None, None, 0, 0, 0, 0, 0, None, None]
    else:
        batch_arr = [None, None, None, None, None, None, 0, 0, 0, None, None]

    if False:
        if np.any(np.array(prior.whiten_space)):
            # whiten transform in space
            Kzz_chol = cholesky(add_jitter(Kzz_full, settings.jitter))
            # TODO: fix the hardcoded time dim
            Kzz_chol = np.kron(np.eye(2), Kzz_chol)
            pred_mean, pred_var_chol =  Kzz_chol @ pred_mean, Kzz_chol @ pred_var_chol
            breakpoint()

    #K_x_t = K_x_t*2

    if settings.avoid_s_cholesky:
        S = pred_var
        spatial_fn = gaussian_linear_operator_spatial_conditional_blocks_avoid_S_chol
    else:
        S = pred_var_chol
        spatial_fn = gaussian_linear_operator_spatial_conditional_blocks

    # batch over time
    mu_p, var_p_bd = jax.vmap(
        spatial_fn,
        batch_arr,
    )( 
        Q*out_dim,
        XS_space, 
        X_space, 
        K_base_spatial_zz, 
        K_spatial_sz, 
        K_spatial_ss, 
        K_x_t, #batching 
        pred_mean, #batching
        S, #batching
        mean_x, 
        mean_xs
    )


    mu_p_bd = np.reshape(mu_p, [-1, Q* out_dim, 1])
    var_p_bd = np.reshape(var_p_bd, [-1, 1, Q*out_dim, Q*out_dim])

    chex.assert_rank([mu_p_bd, var_p_bd], [3, 4])

    return mu_p_bd, var_p_bd


@dispatch(Data, Independent, DifferentialOperatorJoint, FullConjugateGaussian)
def spatial_conditional(
    data_xs, 
    data_x, 
    pred_mean, 
    pred_var, 
    approximate_posterior, 
    likelihood, 
    prior, 
    sparsity,
    out_block_dim, 
    whiten
):

    return differential_spatial_conditional(
        data_xs, 
        data_x, 
        pred_mean, 
        pred_var, 
        approximate_posterior, 
        likelihood, 
        prior, 
        sparsity,
        out_block_dim, 
        whiten
    )

@dispatch(TemporallyGroupedData, Independent, DifferentialOperatorJoint, FullConjugateGaussian)
def spatial_conditional(
    data_xs, 
    data_x, 
    pred_mean, 
    pred_var, 
    approximate_posterior, 
    likelihood, 
    prior, 
    sparsity,
    out_block_dim, 
    whiten
):
    mu, var =  differential_spatial_conditional(
        data_xs, 
        data_x, 
        pred_mean, 
        pred_var, 
        approximate_posterior, 
        likelihood, 
        prior, 
        sparsity,
        out_block_dim, 
        whiten,
        batch_space=True
    )
    return mu, var




@dispatch(Data, DifferentialOperatorJoint, DifferentialOperatorJoint, FullConjugateGaussian)
def spatial_conditional(
    data_xs, 
    data_x, 
    pred_mean, 
    pred_var, 
    aapproximate_posterior, 
    likelihood, 
    prior, 
    sparsity,
    out_block_dim, 
    whiten
):
    """
    Single Latent Function case
    """
    # Wrap prior in an Independent transform so that it is in the same format as the multi latent case
    ind_prior = Independent([prior])

    
    return evoke('spatial_conditional', data_xs, ind_prior, prior, approximate_posterior)(
           data_xs, data_x, pred_mean, pred_var, aapproximate_posterior, likelihood, ind_prior, sparsity, out_block_dim, whiten 
    )



