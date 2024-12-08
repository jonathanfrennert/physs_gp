"""Gaussian likelihood."""
import chex
import objax
import jax
import jax.numpy as np
from jax import grad, jacfwd
from . import Likelihood, FullLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood

from .. import Parameter
from ..core import Block

from ..computation.parameter_transforms import inv_positive_transform, positive_transform
from ..computation.gaussian import log_gaussian_scalar
from ..computation.matrix_ops import vectorized_lower_triangular_cholesky, vectorized_lower_triangular, to_block_diag, batched_diag, mat_inv
from ..computation.integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo

class FullGaussian(FullLikelihood):
    def __init__(self, dim: int = None, variance=None, train=True):
        #if (block_size is None and num_blocks is None) or variance is None:
        #    raise NotImplementedError()

        self.dim = dim

        if variance is None:
            variance = np.eye(dim)

        # TODO: ensure positivity here
        self.variance_param = Parameter(
            variance, 
            constraint=None, 
            name ='FullGaussian/variance', 
            train=True
        )

class BlockDiagonalGaussian(BlockDiagonalLikelihood):
    """
    Block Diagonal Likelihood.


    When used in CVI models we assume that the blocks are ordered in data-latent format. For example:
    
        - Multi-output models with no sparsity will have N blocks of size P x P
        - Multi-output models with spatial sparsity will have Nt blocks of size (Ns P) x (Ns P) ordered in space-latent format

    """
    def __init__(self, block_size:int=None, num_blocks:int=None, num_latents: int = None, variance=None, train=True):
        #if (block_size is None and num_blocks is None) or variance is None:
        #    raise NotImplementedError()

        self._block_size = block_size
        self.num_blocks = num_blocks
        self.num_latents = num_latents

        if variance is None:
            variance = np.tile(np.eye(block_size), [num_blocks, 1, 1])

        chol = vectorized_lower_triangular_cholesky(variance)

        self.variance_param = Parameter(
            variance, 
            inv_constraint_fn = vectorized_lower_triangular_cholesky, 
            constraint_fn = lambda x: vectorized_lower_triangular(x, self.block_size), 
            name ='BlockGaussian/variance', 
            train=True
        )

    def fix(self):
        self.variance_param.fix()

    def release(self):
        self.variance_param.release()

    @property
    def block_size(self):
        return self._block_size

    @property
    def variance(self) -> np.ndarray:
        var_chol =  self.variance_param.value
        # Compute LL^T for each block
        return var_chol @ np.transpose(var_chol, [0, 2, 1])

    @property
    def full_variance(self) -> np.ndarray:
        return to_block_diag(self.variance)

    @property
    def precision(self) -> np.ndarray:
        return jax.vmap(mat_inv)(self.variance)

    @property
    def full_precision(self) -> np.ndarray:
        return to_block_diag(self.precision)

class PrecisionBlockDiagonalGaussian(BlockDiagonalGaussian):
    """ BlockDiagonalGaussian storing the precision not the variance """
    @property
    def variance(self) -> np.ndarray:
        return jax.vmap(mat_inv)(self.precision)

    @property
    def precision(self) -> np.ndarray:
        precision_chol =  self.variance_param.value
        # Compute LL^T for each block
        return precision_chol @ np.transpose(precision_chol, [0, 2, 1])

    @property
    def full_precision(self) -> np.ndarray:
        return to_block_diag(self.precision)

class ReshapedBlockDiagonalGaussian(BlockDiagonalGaussian):
    def __init__(self, bd_lik, block_size:int=None, num_blocks:int=None):
        self.bd_lik = bd_lik
        self._block_size = block_size
        self.num_blocks = num_blocks

    def fix(self):
        self.bd_lik.fix()

    @property
    def base(self):
        return self.bd_lik.base

    @property
    def variance_param(self):
        """ Return the original variance parameter so that it is consistent for gradient updates etc. """
        return self.bd_lik.variance_param

    @property
    def variance(self) -> np.ndarray:
        return batched_diag(np.reshape(self.bd_lik.variance, [self.block_size, self.num_blocks]))

    @property
    def full_variance(self) -> np.ndarray:
        return self.bd_lik.full_variance



class DiagonalGaussian(DiagonalLikelihood):
    """Gaussian likelihood."""

    def __init__(self, variance=None, train=True):

        if variance is None:
            raise NotImplementedError()

        self.variance_param = Parameter(
            np.array(variance), 
            constraint='positive', 
            name ='Gaussian/variance', 
            train=True
        )

    def fix(self):
        self.variance_param.fix()


    @property
    def variance(self) -> np.ndarray:
        return np.diag(np.squeeze(self.variance_param.value))

    @property
    def full_variance(self) -> np.ndarray:
        return self.variance

    def conditional_var(self, f):
        return self.variance

    def conditional_mean(self, f):
        return f

class ReshapedDiagonalGaussian(BlockDiagonalGaussian):
    """ Convert a diagonal Gaussian to a block-block-diagonal matrix """

    def __init__(self, bd_lik, num_outer_blocks:int=None, inner_block_size: int = None, num_latents : int = None):
        self.bd_lik = bd_lik
        self.num_outer_blocks = num_outer_blocks
        self.inner_block_size = inner_block_size
        self.num_latents = num_latents
        self._block_size = self.num_latents*self.inner_block_size

    @property
    def base(self):
        return self.bd_lik.base

    @property
    def variance_param(self):
        """ Return the original variance parameter so that it is consistent for gradient updates etc. """
        return self.bd_lik.variance_param

    @property
    def variance(self) -> np.ndarray:
        var = self.bd_lik.variance_param.value
        Q = self.num_latents

        # convert var to a block diagonal matrix with blocks of size inner_block_size
        var_inner_blocks = np.tile(np.eye(self.inner_block_size)[None, ...], [Q, 1, 1])
        chex.assert_shape(var_inner_blocks, [Q, self.inner_block_size, self.inner_block_size])
        # broad cast multiply with var across the blocks
        var_inner_blocks = var_inner_blocks * var[:, None, None]
        chex.assert_shape(var_inner_blocks, [Q, self.inner_block_size, self.inner_block_size])
        # convert to block diagonal matrix
        var_inner_blocks = to_block_diag(var_inner_blocks)
        chex.assert_shape(var_inner_blocks, [Q*self.inner_block_size, Q*self.inner_block_size])
        # tile across num_outer_blocks
        res = np.tile(var_inner_blocks[None, ...], [self.num_outer_blocks, 1, 1])
        chex.assert_shape(res, [self.num_outer_blocks, Q * self.inner_block_size, Q*self.inner_block_size])

        return res

    @property
    def full_variance(self) -> np.ndarray:
        return self.bd_lik.full_variance

class Gaussian(DiagonalGaussian):
    """Gaussian likelihood."""

    def __init__(self, variance=None):

        if variance is None:
            variance = 1.0

        self.variance_param = Parameter(
            variance, 
            constraint='positive', 
            name ='Gaussian/variance', 
            train=True
        )

    def fix(self):
        self.variance_param.fix()

    def release(self):
        self.variance_param.release()

    @property
    def variance(self) -> np.ndarray:
        return self.variance_param.value

    @property
    def full_variance(self) -> np.ndarray:
        return self.variance

    def log_likelihood_scalar(self, y, f):
        ll = log_gaussian_scalar(y, f, self.variance)
        return ll

    def conditional_var(self, f):
        return self.variance * np.ones_like(f)

    def conditional_mean(self, f):
        return f

    def conditional_samples(self, f, num_samples = 1, generator=None):
        chex.assert_rank(f, 2)
        f = f[..., None]

        samples = mv_indepentdent_monte_carlo(
            lambda x: x,
            self.conditional_mean(f), 
            self.conditional_var(f),
            generator = generator,
            num_samples=num_samples,
            average=False
        )

        # [num_samples, f.shape]
        chex.assert_rank(samples, 4)

        return samples

class ReshapedGaussian(Gaussian):
    def __init__(self, base,  num_blocks, block_size): 
        self._base = base
        self.num_blocks = num_blocks
        self._block_size = block_size

        self.new_mat = np.tile(np.eye(self.block_size), [self.num_blocks, 1, 1])

    def fix(self):
        self._base.fix()

    def release(self):
        self._base.release()

    @property
    def block_size(self):
        return self._block_size

    @property
    def base(self):
        return self._base.base

    @property
    def variance(self) -> np.ndarray:
        return self.new_mat * self.base.variance


class GaussianParameterised(Likelihood):
    """Gaussian likelihood."""

    def __init__(self, kernel):
        self.kernel = kernel

    def variance(self, X) -> np.ndarray:
        return self.kernel.K(X, X)
