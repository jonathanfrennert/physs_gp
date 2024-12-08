from . import Transform, Independent, LinearTransform
from ..computation.permutations import data_order_to_output_order, ld_to_dl, dl_to_ld
from ..computation.matrix_ops import to_block_diag
from ..dispatch import _ensure_str

import jax.numpy as np

class DataStack(Independent):
    """ 
    Stack horizontally across data.

    This inherits Independent as we are doing a mean field assumption (in the prior) across input points.
    """
    def __init__(self, gp: 'Model'):
        assert type(gp) == list
        self._parent = gp
        self._output_dim = 1

    @property
    def latents(self):
        return self.parent.latents

    @property
    def num_latents(self):
        return len(self.latents)

    def get_sparsity_list(self):
        sparsity_list = []

        for p in self.parent:
            sparsity_list += p.get_sparsity_list()

        return [sparsity_list]

class NearestNeighbours(LinearTransform):
    def __init__(self, gp: 'Model'):
        self._parent = gp

    @property
    def latents(self):
        return self.parent.latents

    @property
    def num_latents(self):
        return len(self.latents)

class PrecomputedNearestNeighbours(NearestNeighbours):
    """ For batch GPs this is very inefficiently implemented, and mostly used for debugging."""
    def __init__(self, gp: 'Model', data_group_map = None, groups=None):
        """
        Args:
            groups: how the latent function is grouped across input points
            data_group_map: which gropus to use to compute each data point
        """
        self._parent = gp
        self.data_group_map = data_group_map
        self.groups = groups
        self._output_dim = self.parent.output_dim

    def covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        breakpoint()
        N1 = X1.shape[0]
        N2 = X2.shape[0]
        return self.parent.covar(X1, X2)

    def mean(self, X1: np.ndarray) -> np.ndarray:
        return self.parent.mean(X1)

