""" Covariance Regression Priors """

import jax
import jax.numpy as np
import numpy as onp
import chex


from .transform import Transform, LinearTransform, NonLinearTransform, Independent
from .. import Parameter
from .multi_output import GPRN_DRD
from ..computation.parameter_transforms import get_correlation_cholesky, correlation_transform, inv_correlation_transform,softplus, probit

class WhishartProcess(NonLinearTransform):
    def __init__(self, L, A):
        super(NonLinearTransform, self).__init__()

        self.L = L
        self.A = A

class LKJStaticVarianceProcess(NonLinearTransform):

    def __init__(self, W_vec, a=None, variances =  None, input_dim: int = None, output_dim: int = None):

        self._input_dim = input_dim
        self._output_dim = self.input_dim

        # When using LMC_corr the mixing matrix must be square
        self.P = self.output_dim
        self.Q = int(self.P*(self.P-1)/2)

        if a is None:
            a = 1.0

        self.a = a

        # Set defaults
        if variances is None:
            variances = np.ones(self.P)
        else:
            variances = np.array(variances)

        # Setup Parameters
        self.variances = Parameter(variances, constraint='positive', name='GPRN_DRD/variance', train=True)

        # Flatten latents to fit into VI framework
        self._parent = Independent(
            latents = W_vec,
            prior = True
        )

    def forward(self, f):
        # f has the same ordering as self.latents
        chex.assert_rank(f, 2)

        num_latents = f.shape[0]
        latent_W = np.squeeze(f)
        chex.assert_rank(latent_W, 1)

        correlation_cholesky =  get_correlation_cholesky(
            correlation_transform(latent_W, self.a), 
            self.P, 
            self.Q
        )

        var_diag = np.diag(self.variances.value)

        return (var_diag @ correlation_cholesky @ correlation_cholesky.T @ var_diag.T)

class LKJProcess(NonLinearTransform):
    def __init__(self, v, W_vec, f):
        super(NonLinearTransform, self).__init__()

        self.L = L
        self.A = A


