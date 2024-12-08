""" Helper functions for batching over kernels """

from ..dispatch import dispatch, evoke, _ensure_str
from ..utils.batch_utils import batch_over_module_types
from ..utils.utils import get_batch_type
from .matrix_ops import batched_block_diagional, to_block_diag, add_jitter, cholesky, get_block, mat_inv
from .permutations import permute_vec, permute_mat, data_order_to_output_order
from .. import settings 


import jax
import jax.numpy as np
from jax import jit, vmap, lax
import chex
import objax
from batchjax import batch_or_loop, BatchType

def _batched_diff_kernel(prior, X_time, XS_space, X_space, hierarchical, batch_space=False):
    """ 
    Helper function to compute diff-op time-space kernels separately 

    Prior must be a GP with a spatio-temporal kernel.

    Requires the following format:

    Not Hierarchical:
        prior = Independent: [
            DifferentialOperatorJoint(
                DifferentialOperatorJoint(
                    kern = K_time * K_space,
                    derivative_kernel = Time Only
                )
                derivative_kernel = Space Only
            )
        ]
    """
    q_list = prior.parent

    if batch_space:
        space_batch_arr = [0]
    else:
        space_batch_arr = [None]

    # Helper function to batch across prior
    batch = lambda fn: batch_or_loop(
        fn,
        [q_list],
        [0],
        dim = len(q_list),
        out_dim = 1,
        batch_type = get_batch_type(q_list)
    )

    if batch_space:
        batch_with_xs = lambda fn: batch_or_loop(
            lambda q: jax.vmap(lambda xs: fn(q, xs), [0])(XS_space),
            [q_list],
            [0],
            dim = len(q_list),
            out_dim = 1,
            batch_type = get_batch_type(q_list)
        )
    else:
        batch_with_xs = lambda fn: batch_or_loop(
            lambda q: fn(q, XS_space),
            [q_list],
            [0],
            dim = len(q_list),
            out_dim = 1,
            batch_type = get_batch_type(q_list)
        )

    # returns p(S | T)
    base_prior = lambda q: q.base_prior

    # return K_time
    base_time_kernel = lambda q: base_kern(q).k1

    if batch_space:
        # return K_space
        base_space_kernel = lambda q: base_kern(q).k2
    else:
        # return K_space
        base_space_kernel = lambda q: base_kern(q).k2


    Ms = X_space.shape[0]

    if hierarchical:
        # When hierarchical the base prior is directly the derivative kernel
        base_kern = lambda q: base_prior(q).derivative_kernel.parent_kernel

        # The spatial kernel on the prior does not contain derivaties
        # These are only included in the conditional

        # Ns x Ns
        K_base_spatial_zz_fn = lambda q: base_space_kernel(q).K(X_space, X_space)

        K_spatial_sz_fn = lambda q, xs: (q.covar_from_fn(xs, X_space, base_space_kernel(q).K))[:, :Ms]

        K_x_t_fn = lambda q: jax.vmap(
            lambda t: base_prior(q).covar_from_fn(t, t, base_time_kernel(q).K)
        )(X_time[:, None, :])

        K_spatial_ss_fn = lambda q, xs: jax.vmap(
            lambda x: q.covar_from_fn(x[None, :], x[None, :], base_space_kernel(q).K)
        )(xs)

    else:
        # return K_time * K_space
        base_kern = lambda q: base_prior(q).parent.derivative_kernel.parent_kernel

        # Compute \nabla K_space 
        K_base_spatial_zz_fn = lambda q: base_prior(q).covar_from_fn(X_space, X_space, base_space_kernel(q).K)

        K_spatial_sz_fn = lambda q, xs: base_prior(q).covar_from_fn(xs, X_space, base_space_kernel(q).K) 

        K_x_t_fn = lambda q: jax.vmap(
            lambda t: base_prior(q).parent.covar_from_fn(t, t, base_time_kernel(q).K)
        )(X_time[:, None, :])

        K_spatial_ss_fn = lambda q, xs: jax.vmap(
            lambda x: base_prior(q).covar_from_fn(x[None, :], x[None, :], base_space_kernel(q).K)
        )(xs)



    # N referes to the number of spatial points

    # [Q] x [Ds x N] x [Ds x N]
    K_base_spatial_zz_arr = batch(K_base_spatial_zz_fn)

    # [Q] x [Ds x Ns] x [Ds x N]
    K_spatial_sz_arr = batch_with_xs(K_spatial_sz_fn)

    # [Q] x [Ns] x [Ds] x [Ds]
    K_base_spatial_ss_arr = batch_with_xs(K_spatial_ss_fn)

    #[Q] x [Nt] x [Dt] x [Dt]
    K_x_t_arr= batch(K_x_t_fn)

    if batch_space:

        # K_base_spatial_ss_arr has shape [Q] x [Nt] x [Ns] x [Ds] x [Ds]
        # K_spatial_sz_arr has shape [Q] x [Nt] x [DsxNs]x[N]

        # Fix shapes
        # K_base_spatial_ss_arr -> [Nt] x [Ns] x [Q] x [Ds] x [Ds]
        K_base_spatial_ss_arr = np.transpose(K_base_spatial_ss_arr, [1, 2, 0, 3, 4])

        # K_spatial_sz_arr -> [Nt]xQx[DsxNs]x[DsxN] or [Nt]xQx[DsxNs]x[N]
        K_spatial_sz_arr = np.transpose(K_spatial_sz_arr, [1, 0, 2, 3])
    else:
        # Fix shapes
        # [Ns] x [Q] x [Ds] x [Ds]
        K_base_spatial_ss_arr = np.transpose(K_base_spatial_ss_arr, [1, 0, 2, 3])

    #[Nt] x [Q] x [ Dt] x [ Dt]
    K_x_t_arr = np.transpose(K_x_t_arr, [1, 0, 2, 3])


    return K_x_t_arr, K_base_spatial_ss_arr, K_spatial_sz_arr, K_base_spatial_zz_arr

def _batched_st_kernel(X1, X2, prior, kernel_type='spatial', full=True):
    """ 
    Helper function to compute time-space kernels separately 

    Prior must be a GP with a spatio-temporal kernel.
    """

    if kernel_type == 'spatial':
        if full:
            fn = lambda q: q.kernel.k2.K(X1, X2)
        else:
            fn = lambda q: q.kernel.k2.K_diag(X1)
    elif kernel_type == 'temporal':
        if full:
            fn = lambda q: q.kernel.k1.K(X1, X2)
        else:
            fn = lambda q: q.kernel.k1.K_diag(X1)
    else:
        raise NotImplementedError()

    # TODO: .latents is depreciated
    q_list = prior.latents

    K_arr = batch_or_loop(
        fn,
        [q_list],
        [0],
        dim = len(q_list),
        out_dim = 1,
        batch_type = get_batch_type(q_list)
    )

    return K_arr

