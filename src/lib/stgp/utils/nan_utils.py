"""
JIT requires that matrix sizes are known. Although masking can be done in advance, it is more convenient to allowing dyanmic masking. This has the side affect that a mask does not need to be passed around so that the resulting framework is cleaener.
"""

import jax
import jax.numpy as np
import objax
import chex
from jax import jit

from ..computation.matrix_ops import cholesky, cholesky_solve, add_jitter, first_axis_dim

def get_same_shape_mask(Y: np.ndarray) -> np.ndarray:
    """
    Returns 1 if Y_n is numeric, otherwise 0
    """
    if type(Y) is list:
        return [get_same_shape_mask(y) for y in Y]
    else:
        return (~np.isnan(Y)).astype(int)

def get_mask(Y: np.ndarray) -> np.ndarray:
    """
    Returns 1 if Y_n is numeric, otherwise 0
    """
    if type(Y) is list:
        return [get_mask(y) for y in Y]
    else:
        mask =  get_same_shape_mask(Y)
        # make sure rank one
        return np.reshape(mask, [mask.shape[0]])

@jit
def mask_vector(Y, mask):
    chex.assert_equal(Y.shape[0], mask.shape[0])
    chex.assert_rank(Y, 2)
    chex.assert_rank(mask, 1)

    return np.where(mask[:, None], Y, np.zeros_like(Y))

def mask_matrix(Y, mask):
    if type(Y) is list:
        return [mask_matrix(Y[i], mask[i]) for i in range(first_axis_dim(Y))]
    else:
        chex.assert_equal(Y.shape, mask.shape)
        return np.where(mask, Y, np.zeros_like(Y))


@jit
def mask_to_identity(K: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    K is (probably) a full p.s.d matrix. We want to decorrelate any ... tbd 
    """

    chex.assert_rank(K, 2)
    chex.assert_rank(mask, 1)
    chex.assert_equal(K.shape[0], mask.shape[0])
    chex.assert_equal(K.shape[1], mask.shape[0])

    N = K.shape[0]

    mask = np.tile(mask, [mask.shape[0], 1]) 

    K = K-np.eye(N)
    K = np.multiply(K, mask)
    K = np.multiply(K, mask.T)
    K = K+np.eye(N)

    return K

def get_diag_mask(mask: np.ndarray) -> np.ndarray:
    return np.diag(mask)
