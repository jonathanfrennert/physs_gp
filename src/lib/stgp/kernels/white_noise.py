from . import Kernel
import jax
import jax.numpy as np
import chex

class WhiteNoise(Kernel):
    def K_diag(self, X1):
        return np.ones(X1.shape[0])

    def _K(self, X1, X2):

        if X1.shape[0] == X2.shape[0]:
            return np.eye(X1.shape[0])

        return np.zeros([X1.shape[0], X2.shape[0]])

        x_eq = np.equal(X1, X2).astype(int)
        K =  np.diagflat(np.prod(x_eq, axis=1))

        chex.assert_equal(K.shape[0], X1.shape[0])
        chex.assert_equal(K.shape[1], X2.shape[0])

        return K

