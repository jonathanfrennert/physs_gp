import jax
import jax.numpy as np
from jax import jit
from jax.scipy.special import erf, gammaln
from jax.scipy.stats.norm import logcdf

import chex

@jit
def log_poisson(x, lam):
    return x * np.log(lam) - lam - gammaln(x + 1.0)

@jit
def log_bernoulli(y, p):
    """ 
    Bernoulli is given as:
        p^y * (1-p)*(1-y)

    where p is in [0, 1]
    """
    chex.assert_rank([y, p], [0, 0])

    # add jit to avoid log of zero
    jitter = 1e-5

    return y * np.log(p + jitter) + (1-y) * np.log(1 - p + jitter)


@jit
def log_inv_probit(f):
    return logcdf(f)


