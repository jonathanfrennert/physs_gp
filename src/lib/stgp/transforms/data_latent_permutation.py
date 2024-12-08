"""
Converts a prior from latent-data format to data-latent format.  This is required when using a FullApproximatePosterior 

There are two separate classes and we can exploit sparsity when permuting independent priors.
"""
import jax
import jax.numpy as np
from batchjax import batch_or_loop, BatchType
import chex
from functools import partial

from . import Transform, LinearTransform
from ..computation.matrix_ops import block_from_mat, v_get_block_diagonal, to_block_diag, get_block_diagonal
from ..computation.permutations import data_order_to_output_order, permute_vec_blocks, permute_vec, permute_mat, lp_blocks, permute_blocks, left_permute_mat, right_permute_mat
from ..utils.utils import ensure_module_list, get_batch_type


class DataLatentPermutation(LinearTransform):
    def __init__(self, latents):

        # Allow passing a list of prior models and transformed model
        if type(latents) is list:
            self._parent = Independent(latents=latents, prior=True)
        else:
            self._parent = latents 

        self._output_dim = self.parent.output_dim
        self._input_dim = self.parent.input_dim

    def np_mean_blocks(self, X): 
        """ Computes the mean across all latents keeping X fixed """
        #chex.assert_rank(X, 2)
        return self.parent.mean_blocks(X)

    def np_b_mean_blocks(self, X):
        """ 
        X is of rank 3, one X per latent function.
        This computes the mean of each latent function with its corresponding X
        """
        #chex.assert_rank(X, 3)
        return self.parent.b_mean_blocks(X)

    def permute_vec(self, vec, Q):
        return permute_vec(vec, Q)

    def permute_mat(self, mat, Q):
        return permute_mat(mat, Q)

    def mean(self, X):
        """ Return mean in data-latent format """
        return permute_vec_blocks(self.np_mean_blocks(X))

    def b_mean(self, X):
        """ Return (batched X) mean in data-latent format """
        return permute_vec_blocks(self.np_b_mean_blocks(X))

    def np_mean(self, X):
        """ mean without permutations """
        return self.parent.mean(X)

    def np_rb_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """ X1 is static. No Permutations """

        # TODO: this is a hack, for efficieny parent should support this natively
        X1 = np.tile(X1, [X2.shape[0], 1, 1])

        return self.parent.b_covar(X1, X2)

    def np_b_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """ No Permutations """

        return self.parent.b_covar(X1, X2)

    def np_covar_blocks(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """ X1 and X2 are static. No Permutations """

        return self.parent.covar_blocks(X1, X2)

    def np_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """ X1 and X2 are static. No Permutations """
        return self.parent.covar(X1, X2)

    def lp_rb_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Left Permute, keep X1 static when constructing full var
        """
        # TODO: this is a hack, for efficieny parent should support this
        X1 =np.repeat(X1[None, ...], X2.shape[0], axis=0)

        return left_permute_mat(self.parent.b_covar(X1, X2), self.output_dim)

    def lp_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Left Permute, keep both x1 and x2 static when constructing full var
        """
        return lp_blocks(self.np_covar_blocks(X1, X2))


    def covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Permute, static X1 and X2
        """
        #return permute_blocks(self.np_covar_blocks_blocks(X1, X2))
        return permute_mat(self.parent.covar(X1, X2), self.output_dim)

    def b_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Permute, static X1 and X2
        """
        #return permute_blocks(self.np_covar_blocks_blocks(X1, X2))
        return permute_mat(self.parent.b_covar(X1, X2), self.output_dim)

    def var(self, X1: np.ndarray) -> np.ndarray:
        """
        Compute permuted diagonal variance vector
        """
        K = self.parent.var(X1)
        K = np.vstack(K)
        return self.permute_vec(K)

    def full_var(self, X1: np.ndarray) -> np.ndarray:
        """
        Compute permuted full covariance keeping X1 static
        """
        return self.covar(X1, X1)

    def full_var_blocks(self, X: np.ndarray, group_size, block_size) -> np.ndarray:
        """
        Compute block diagonal of permuted full var whilst keeping X static
        """
        raise NotImplementedError()

    def b_full_var_blocks(self, X: np.ndarray, group_size, block_size) -> np.ndarray:
        full_var = self.b_covar(X, X)
        return get_block_diagonal(full_var, block_size)

    def _b_full_var_blocks(self, X: np.ndarray, group_size, block_size) -> np.ndarray:
        """
        Compute block diagonal of permuted full var whilst batching X

        X is a grouped input matrix:
            [N_g, S_g, D]
        group_size is required data_grouping
        block_size is the size variance for the corresponding groups
        """
        # Group data

        X = jax.vmap(
            block_from_mat,
            [0, None],
            0
        )(X, group_size)

        X = np.transpose(X, [1, 0, 2, 3])

        # For each group collect blocks
        K_blocks = jax.vmap(
            lambda m, x: m.b_covar(x, x),
            [None, 0],
            0
        )(self, X)

        K_blocks = v_get_block_diagonal(
            K_blocks,
            block_size,
            K_blocks.shape[1]
        )


        return K_blocks

class IndependentDataLatentPermutation(DataLatentPermutation):
    pass

class IndependentJointDataLatentPermutation(DataLatentPermutation):
    pass

class JointDataLatentPermutation(DataLatentPermutation):
    """
    Converts a prior from latent-data format to data-latent format.
    This is required when using a FullApproximatePosterior .

    Assumes that parent is a full prior

    Function name syntax:
        p: permute
        lp: left permute
        rp: left permute
        np: no permute
        s: static (ie the input should not be batched over)
        b: batched (ie the input should be batched over)
    """
    def __init__(self, latents):

        super(JointDataLatentPermutation, self).__init__(latents)

        self._p = latents

        self._output_dim = self.parent.output_dim 
        self._input_dim = self.parent.input_dim

    def forward(self, *args, **kwargs): 
        return self.parent.forward(*args, **kwargs)


