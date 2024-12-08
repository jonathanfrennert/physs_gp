""" utils for converting from latent-data to data-latent format """
import jax
import jax.numpy as np
from jax import jit
from functools import partial
import chex

from .matrix_ops import to_block_diag




#@partial(jit, static_argnums=(0, 1))
def data_order_to_output_order(num_outputs: int, N: int):
    """
    A permutation matrix to convert from latent-data to data-latent order

    For example if  we have a vector in Y = [data-latent] format with N-P
    then we can convert to [latent-data] format by
        data_order_to_output_order(P, N).T @ Y
    """
    total_N = N*num_outputs

    i = np.hstack([np.arange(i , total_N, N) for i in range(N)])
    permutation = np.eye(total_N)[i]

    return permutation

@jit
def permute_vec_blocks(v_blocks):
    """
    Converts blocks to a permuted vector.

        v_blocks is of rank 3 where the first axis are latents, the second is data

        The default ordering in numpy is C-like/row major which stacks rows. However we need to stack columns (latents)
            so we use F-like / column major ordering.

        So to convert to data - latent order we simply need to reshape.
    """

    chex.assert_rank(v_blocks, 3)

    return np.reshape(v_blocks, [-1, v_blocks.shape[-1]], order='F')

@partial(jit, static_argnums=(1))
def permute_vec(v, num_latents):
    """ Permute vec and return vec"""

    # TODO: write test
    N = int(v.shape[0]/num_latents)
    return np.reshape(np.reshape(v, [num_latents, N]).T, [-1, 1])

    chex.assert_rank(v, 2)
    P = data_order_to_output_order(
        num_latents,
        int(v.shape[0]/num_latents)
    )

    return P @ v

@partial(jit, static_argnums=(1))
def unpermute_vec(v, num_latents):
    """ Unpermute vec and return vec"""
    chex.assert_rank(v, 2)
    P = data_order_to_output_order(
        num_latents,
        int(v.shape[0]/num_latents)
    )

    return P.T @ v

@jit
def lp_blocks(K_blocks):
    """ Compute left permtued full matrix from blocks """
    chex.assert_rank(K_blocks, 3)

    Q = K_blocks.shape[0]
    N1 = K_blocks.shape[1]

    K = to_block_diag(K_blocks)

    return np.vstack(np.transpose(np.reshape(K, [Q, N1, -1]), [1, 0, 2]))

@partial(jit, static_argnums=(1))
def permute_blocks(A_blocks, num_latents):
    chex.assert_rank(A_blocks, 3)
    lp_A = lp_blocks(A_blocks) 

    right_P = data_order_to_output_order(
        num_latents,
        int(lp_A.shape[-1]/num_latents)
    )

    return lp_A @ right_P.T

@partial(jit, static_argnums=(1))
def left_permute_mat(A, num_latents):
    chex.assert_rank(A, 2)
    if True:
        # TODO: write test
        N = int(A.shape[0]/num_latents)
        return np.reshape(
            np.transpose(
                np.reshape(A, [num_latents, N, A.shape[1]]),
                [1, 0, 2]
            ),
            [-1, A.shape[1]]
        )

    left_P = data_order_to_output_order(
        num_latents,
        int(A.shape[0]/num_latents)
    )

    return left_P @ A 

@partial(jit, static_argnums=(1))
def right_permute_mat(A, num_latents):

    return left_permute_mat(A.T, num_latents).T
    chex.assert_rank(A, 2)

    right_P = data_order_to_output_order(
        num_latents,
        int(A.shape[1]/num_latents)
    )

    return A @ right_P.T 

@partial(jit, static_argnums=(1))
def permute_mat(A, num_latents):
    chex.assert_rank(A, 2)

    A = left_permute_mat(A, num_latents)

    return right_permute_mat(A, num_latents)

@partial(jit, static_argnums=(1))
def unpermute_mat(A, num_latents):
    chex.assert_rank(A, 2)

    left_P = data_order_to_output_order(
        num_latents,
        int(A.shape[0]/num_latents)
    )

    right_P = data_order_to_output_order(
        num_latents,
        int(A.shape[1]/num_latents)
    )

    return left_P.T @ A @ right_P

def ld_to_dl(num_latents: int, num_data: int):
    """ Create permutation matrix to convert from latent-data to data-latent """
    return data_order_to_output_order(num_latents, num_data)

def dl_to_ld(num_latents: int, num_data: int):
    """ Create permutation matrix to convert from data-latent to latent-data """

    return data_order_to_output_order(num_latents, num_data).T

def permute_mat_ld_to_dl(mat, num_latents: int, num_data: int):
    P = ld_to_dl(num_latents, num_data)
    return P @ mat @ P.T

def permute_vec_ld_to_dl(vec, num_latents: int, num_data: int):
    P = ld_to_dl(num_latents, num_data)
    return P @ vec 

def permute_mat_dl_to_ld(mat, num_latents: int, num_data: int):
    P = dl_to_ld(num_latents, num_data)
    return P @ mat @ P.T

def permute_vec_dl_to_ld(vec, num_latents: int, num_data: int):
    # data latent to latent data
    P = dl_to_ld(num_latents, num_data)
    return P @ vec 

def permute_vec_tps_to_tsp(vec, num_latents:int):
    # time-latent-space to time-space-latent
    Nt = vec.shape[0]
    Ns = int(vec.shape[1]/num_latents)
    return jax.vmap(lambda A: permute_vec_ld_to_dl(A, num_latents, Ns))(vec)

def permute_mat_tps_to_tsp(vec, num_latents:int):
    # time-latent-space to time-space-latent
    chex.assert_rank(vec, 3)
    Nt = vec.shape[0]
    Ns = int(vec.shape[1]/num_latents)
    return jax.vmap(lambda A: permute_mat_ld_to_dl(A, num_latents, Ns))(vec)

def permute_mat_tsp_to_tps(vec, num_latents:int):
    # time-space-latent to time-latent-space
    chex.assert_rank(vec, 3)
    Nt = vec.shape[0]
    Ns = int(vec.shape[1]/num_latents)
    return jax.vmap(lambda A: permute_mat_dl_to_ld(A, num_latents, Ns))(vec)

