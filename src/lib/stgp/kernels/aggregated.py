from .kernel import Kernel

import jax
import jax.numpy as np
import chex

class AggregatedKernel(Kernel):
    def __init__(
        self,
        parent_kernel,
    ):
        super(AggregatedKernel, self).__init__()
        self.parent_kernel = parent_kernel


    def K_diag(self, X: np.array):
        chex.assert_rank([X], [3])
        X = self._apply_active_dim(X)

        def _K_d1(x1):
            N = x1.shape[0]
            return (1/(N*N))*np.sum(self.parent_kernel.K(x1, x1))

        K = jax.vmap(_K_d1, in_axes=[0], out_axes=0)(X)

        return K

    def _K_img(self, X1, X2):
        N1 = X1.shape[0]
        N2 = X2.shape[0]
        return (1/(N1*N2))*np.sum(self.parent_kernel.K(X1, X2))

    def _K(self, X1, X2):
        D = X1.shape[1]
        def _K_d2(x1, x2):
            k_d1_d2 = self._K_img(x1, x2)
            chex.assert_rank(k_d1_d2, 0)
            return k_d1_d2

        def _K_d1(x1, X2):
            #vectorised over first input
            return jax.vmap(_K_d2, in_axes=[None, 0], out_axes=0)(x1, X2)

        K = jax.vmap(_K_d1, in_axes=[0, None], out_axes=0)(X1, X2)

        chex.assert_equal(K.shape[0], X1.shape[0])
        chex.assert_equal(K.shape[1], X2.shape[0])

        return K

    def K(self, X1: np.array, X2: np.array):
        chex.assert_rank([X1, X2], [3, 3])
        chex.assert_equal(X1.shape[2], X2.shape[2])

        _X1 = self._apply_active_dim(X1)
        _X2 = self._apply_active_dim(X2)

        K =  self._K(_X1, _X2)

        chex.assert_rank(K, 2)
        chex.assert_equal(K.shape[0], X1.shape[0])
        chex.assert_equal(K.shape[1], X2.shape[0])

        return K


