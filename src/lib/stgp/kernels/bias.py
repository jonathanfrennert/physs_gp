from .kernel import Kernel
import jax.numpy as np

class BiasKernel(Kernel):
    def K_diag(self, X1):
        N = X1.shape[0]
        return np.ones(N)

    def _K(self, X1, X2):
        N1 = X1.shape[0]
        N2 = X2.shape[0]

        return np.ones([N1, N2])
