from . import StationaryKernel
import jax
import jax.numpy as np
import chex

from jax import jit
from functools import partial

class RBF(StationaryKernel):
    def __init__(self, *args, **kwargs):
        super(RBF, self).__init__(*args, **kwargs, name='RBF')

    @partial(jit, static_argnums=(0))
    def _K_scaler(self, x1, x2, lengthscale):
        #ensure scalar inputs
        chex.assert_rank(x1, 0)
        chex.assert_rank(x2, 0)
        chex.assert_rank(lengthscale, 0)

        return  np.exp(-0.5*((x1-x2)**2)/(lengthscale**2))



    
