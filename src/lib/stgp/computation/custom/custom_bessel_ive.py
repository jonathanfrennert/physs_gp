from functools import partial
from jax import jit, grad
import jax.numpy as np
import numpy as onp
import os

from interpax import interp1d


@partial(jit, static_argnums=(0, 2))
def custom_bessel_ive(order, x, interp):
    """ Approximate modified Bessel function of the first kind """

    raise RuntimeError()
    
    

