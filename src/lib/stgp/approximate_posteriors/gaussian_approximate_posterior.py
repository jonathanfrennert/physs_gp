import jax.numpy as np
import numpy as onp
import objax

from . import ApproximatePosterior
from ..computation.matrix_ops import lower_triangular_cholesky, lower_triangle, diagonal_from_cholesky
from .. import Parameter
import chex
import warnings

class GaussianApproximatePosterior(ApproximatePosterior):
    def __init__(self, dim: int=None, m=None, S=None, S_chol_vec = None, train=True, num_latents = None, approximate_posteriors = None):
        super(GaussianApproximatePosterior, self).__init__()

        if dim is None and m is None and approximate_posteriors is None:
            raise RuntimeError('Either dim, m, or approximate_posterior must be passed')

        # this is a bit of hack to get MeanField and FullGaussian approximate posteriors to handle meanfield data posteriors consistently
        if approximate_posteriors is not None:
            self._m = None
            self._S_chol = None
            self.dim = None
            self.num_latents = None
            self.approximate_posteriors = None
            self.meanfield_over_data = True

        else:
            self.approximate_posteriors = None
            self.meanfield_over_data = False

            if m is None:
                #m = 0.01*np.ones([dim, 1])
                m = np.array(0.01*onp.random.rand(dim)[:, None])
                #m = np.ones([dim, 1])

            chex.assert_rank(m, 2)

            if S is None and S_chol_vec is None:
                warnings.warn('Approximate posterior ')
                S = 0.1*np.eye(dim)

            if dim is None:
                dim = m.shape[0]

            self._m = Parameter(
                np.array(m),
                constraint=None,
                name='GaussianApproxPosterior/m',
                train=train
            )

            if S_chol_vec is None:
                S_chol_vec = lower_triangular_cholesky(S)

            self._S_chol = Parameter(
                np.array(S_chol_vec),
                constraint=None,
                name='GaussianApproxPosterior/S_chol',
                train=train
            )

            self.dim = dim
            self.num_latents = num_latents

    @property
    def m(self):
        return self._m.value

    @property
    def S_chol(self):
        S_chol_raw = self._S_chol.value

        return lower_triangle(S_chol_raw, self.dim)

    @property
    def S(self):
        S_chol = self.S_chol
        return S_chol @ S_chol.T

    @property
    def S_diag(self):
        return diagonal_from_cholesky(self.S_chol)

    def fix(self):
        self._m.fix()
        self._S_chol.fix()

    def release(self):
        self._m.release()
        self._S_chol.release()

class DiagonalGaussianApproximatePosterior(GaussianApproximatePosterior):
    def __init__(self, dim: int=None, m=None, S_diag=None, train=True):

        if dim is None and m is None:
            raise RuntimeError('Either dim or m must be passed')

        if m is None:
            #m = 0.01*np.ones([dim, 1])
            m = np.array(0.01*onp.random.rand(dim)[:, None])
            #m = np.ones([dim, 1])
        else:
            m = np.array(m)

        chex.assert_rank(m, 2)

        if S_diag is None:
            warnings.warn('Approximate posterior ')
            S_diag = 0.1*np.ones(dim)
        else:
            S_diag = np.array(S_diag)

        chex.assert_rank(S_diag, 1)


        self._m = Parameter(
            m,
            constraint=None,
            name='DiagonalGaussianApproximatePosterior/m',
            train=train
        )

        self._S_diag = Parameter(
            S_diag,
            constraint=None,
            name='DiagonalGaussianApproximatePosterior/S_diag',
            train=train
        )

    @property
    def m(self):
        return self._m.value

    @property
    def S_diag(self):
        return np.square(self._S_diag.value)

    @property
    def S_chol(self):
        S_diag = self._S_diag.value
        return np.diag(S_diag)

    @property
    def S(self):
        S_chol = self.S_chol
        return S_chol @ S_chol.T

    def fix(self):
        self._m.fix()
        self._S_diag.fix()

    def release(self):
        self._m.release()
        self._S_diag.release()


class FullGaussianApproximatePosterior(GaussianApproximatePosterior):
    """ Gaussian approximate posterior defined in latent-data format """
    pass

class DataLatentBlockDiagonalApproximatePosterior(FullGaussianApproximatePosterior):
    """ 
    Block diagonal approximate posterior that is parameterised in data-latent order.
    This inherits FullGaussianApproximatePosterior as it is used by CVI in the FullGaussianApproximatePosterior
    """
    def __init__(self,  m=None, S_blocks=None, train=True):
        self._m = Parameter(
            m,
            constraint=None,
            name='GaussianApproxPosterior/m',
            train=train
        )

        self._S_blocks = Parameter(
            S_blocks,
            constraint=None,
            name='GaussianApproxPosterior/S_blocks',
            train=train
        )

    @property
    def m(self):
        return self._m.value

    @property
    def S_blocks(self):
        return self._S_blocks.value
