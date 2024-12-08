import objax
import chex
import jax
import jax.numpy as np

from . import StationaryVarianceKernel
from typing import List, Optional, Union
from ..computation.parameter_transforms import inv_positive_transform, positive_transform
from .. import Parameter


class SM_Component(StationaryVarianceKernel):
    """
        We re-use lengthscale to mean \mu
                   variance to be v
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        input_dim: Optional[int] = 1,
        active_dims: Optional[np.ndarray] = None,
    ) -> None:

        super(SM_Component, self).__init__(
            lengthscales=mu,
            variance=v,
            input_dim = input_dim, 
            active_dims = active_dims, 
        )

    def _K_scaler_with_var(self, x1, x2, mu, v):
        #ensure scalar inputs
        chex.assert_rank(x1, 0)
        chex.assert_rank(x2, 0)
        chex.assert_rank(mu, 0)
        chex.assert_rank(v, 0)


        tau = x1-x2
        tau2 = tau**2

        return np.exp(-2*np.pi*tau2*v)*np.cos(2*np.pi*tau*mu)
