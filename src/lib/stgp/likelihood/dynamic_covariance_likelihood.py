import objax
import jax.numpy as np
import chex


from . import Likelihood, DiagonalLikelihood
from ..computation.gaussian import log_gaussian

class DynamicCovarianceLikelihood(DiagonalLikelihood):
    pass

class DynamicCovarianceGaussian(DynamicCovarianceLikelihood):
    def log_likelihood_scalar(self, y, f):
        chex.assert_rank([y, f], [1, 2])
        chex.assert_equal(y.shape[0], f.shape[0])
        Y = y[:, None]

        return log_gaussian(Y, np.zeros_like(Y), f)
