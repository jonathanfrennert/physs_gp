from . import LinearTransform
from ..utils.utils import ensure_module_list, can_batch, get_batch_type
from batchjax import batch_or_loop, BatchType
from ..computation.matrix_ops import to_block_diag, batched_diagonal_from_XDXT

import jax
import jax.numpy as np
import objax
import chex

from typing import List, Optional

class _OutputMap(LinearTransform):
    def __init__(self, parent, mapping, data_y_index=None):
        self._parent = parent
        self.latents = self.parent.latents
        self.mapping = np.array(mapping)
        self._output_dim = len(mapping)
        self.data_y_index = None

    def forward(self, f):
        return f[self.mapping]

    def transform(self, base_mu, base_var):
        chex.assert_rank([base_mu, base_var], [2, 3])

        t_mu = base_mu[self.mapping]
        t_var = base_var[:, self.mapping, :]
        t_var = t_var[:, :, self.mapping]

        return t_mu, t_var

    def transform_diagonal(self, base_mu, base_var):
        P, A = base_mu.shape

        if A != 1: raise NotImplementedError()

        t_mu = base_mu[self.mapping, :]
        t_var = base_var[self.mapping, ...]

        return t_mu, t_var

    def mean_blocks(self, X):
        """ Inefficient implementation for compatability """
        # compute mean from parent
        # will be shape P x N x 1
        parent_mean = self.parent.mean_blocks(X)

        raise NotImplementedError()

        # only select and return the correct outputs
        sub_mean = parent_mean[self.mapping, ...]

        chex.assert_rank(sub_mean, 3)

        return sub_mean

    def mean(self, X):
        """ Inefficient implementation for compatability """

        sub_mean = self.mean_blocks(X)
        sub_mean = np.vstack(sub_mean)

        chex.assert_rank(sub_mean, 2)

        return sub_mean

    def covar(self, X1, X2):
        """ Inefficient implementation for compatability """

        # compute full covariance of parent
        # will be P * N1 x P * N2
        parent_covar = self.parent.covar(X1, X2)
        raise NotImplementedError()

        N1 = X1.shape[0]
        N2 = X2.shape[0]
        P = self.parent.output_dim

        # covar is organised in output - latent format
        # To index into each output we create ranges up to N1 for each output
        # and then shift each range to its corresponding output
        left_idx = np.arange(N1)
        left_idx = np.tile(left_idx, [self.output_dim, 1])
        left_idx = left_idx + self.mapping[:, None] * N1
        left_idx = np.hstack(left_idx)
        chex.assert_shape(left_idx, [N1 * self.output_dim])

        # repeat for right_idx
        right_idx = np.arange(N2)
        right_idx = np.tile(right_idx, [self.output_dim, 1])
        right_idx = right_idx + self.mapping[:, None] * N2
        right_idx = np.hstack(right_idx)
        chex.assert_shape(right_idx, [N2 * self.output_dim])

        # index
        sub_covar = parent_covar[left_idx, :]
        sub_covar = sub_covar[:, right_idx]

        #ensure correct rank
        chex.assert_shape(sub_covar, [N1 * self.output_dim, N2 * self.output_dim])

        return sub_covar

    def covar_blocks(self, X1, X2):
        """ Inefficient implementation for compatability """

        # compute covar blocks from parent
        # will be shape P x N x N
        parent_covar_blocks = self.parent.covar_blocks(X)

        # only select and return the correct outputs
        sub_covar_blocks = parent_covar_blocks[self.mapping, ...]

        chex.assert_rank(sub_covar_blocks, 3)

        return sub_covar_blocks


class OutputMap(LinearTransform):
    """
    Map Outputs
    """
    def __new__(cls, parent, mapping: list):
        # construct a new mapping object for each required map
        obj_list = []
        for m_arr in mapping:
            obj_list.append(
                _OutputMap(parent, m_arr)
            )

        if len(obj_list) == 1:
            return obj_list[0]

        return obj_list

