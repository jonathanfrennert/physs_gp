from . import StationaryVarianceKernel
import jax
import jax.numpy as np
import chex

class RQ(StationaryVarianceKernel):
    def K_diag(self, X1):
        return np.power((1),-self.lengthscale) * np.ones(X1.shape[0])
    def _K_scaler_with_var(self, x1, x2, lengthscale, variance):
        #ensure scalar inputs
        chex.assert_rank(x1, 0)
        chex.assert_rank(x2, 0)
        chex.assert_rank(lengthscale, 0)
        chex.assert_rank(variance, 0)

        r = (x1-x2)**2
        return  np.power(
            1+r/(2*lengthscale*variance),
            - lengthscale
        )




