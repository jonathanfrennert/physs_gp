"""
(Joint) Species Models.

Many of these can be recovered in through multi_output however we can keep them explicit here.
"""

from .transform import Transform, LinearTransform, NonLinearTransform, Independent

import typing
from typing import List, Optional, Union
import jax
import jax.numpy as np
import numpy as onp
import objax
import chex


class AdditiveSpeciesModel(LinearTransform):
    """
    Explictely captues the residual error with a spatio(-temporal) GP.
    """

    def __init__(self, gp, residual_gp):
        self._parent = Independent(
            [gp, residual_gp],
            prior = True
        )

        self._input_dim = 2
        self._output_dim = 1

    def mean(self, X):
        b_mean = self.parent.mean_blocks(X)
        return np.sum(b_mean, axis=0)

    def covar(self, X1, X2):
        b_var = self.parent.covar_blocks(X1, X2)
        return np.sum(b_var, axis=0)

    def var(self, X):
        b_var = self.parent.var_blocks(X)
        return np.sum(b_var, axis=0)

    def full_var(self, X):
        b_var = self.parent.full_var_blocks(X)
        return np.sum(b_var, axis=0)

    def forward(self, f):
        """
        Args:
            f[0] = gp
            f[1] = residual_gp
        """

        return f[0] + f[1]

class AdditiveSpeciesModelWithLMCResidual(LinearTransform):
    pass

class AdditiveSpeciesModelWithGPRNResidual(NonLinearTransform):
    pass

class AdditiveLMCSpeciesModel(LinearTransform):
    pass

class AdditiveGPRNSpeciesModel(LinearTransform):
    pass



