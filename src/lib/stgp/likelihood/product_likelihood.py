import jax.numpy as np
import objax
from batchjax import batch_or_loop, BatchType

from . import Likelihood, Gaussian, BlockDiagonalGaussian
from ..utils.utils import ensure_module_list, can_batch, get_batch_type
from ..core import Block

def get_product_likelihood(likelihood_arr):
    if all(type(lik) == Gaussian for lik in likelihood_arr):
        return GaussianProductLikelihood(likelihood_arr)

    if all(issubclass(type(lik), BlockDiagonalGaussian) for lik in likelihood_arr):
        return BlockGaussianProductLikelihood(likelihood_arr)

    return ProductLikelihood(likelihood_arr)


class ProductLikelihood(Likelihood):
    def __init__(self, likelihood_arr):
        super(ProductLikelihood, self).__init__()
        self.likelihood_arr = objax.ModuleList(likelihood_arr)

    @property
    def block_type(self):
        # TODO: assuming that all product likelihoods 
        return self.likelihood_arr[0].block_type

    def fix(self):
        for lik in self.likelihood_arr:
            lik.fix()

    def release(self):
        for lik in self.likelihood_arr:
            lik.release()


    def log_likelihood_scalar(self, y, f):
        if len(self.likelihood_arr) == 1:
            # just pass through
            return self.likelihood_arr[0].log_likelihood_scalar(y, f)

        ll_arr = []
        for i, lik in enumerate(self.likelihood_arr):
            ll_arr.append(
                lik.log_likelihood_scalar(y[i], f[i])
            )

        return np.array(ll_arr)

    def conditional_var(self, f):
        var_arr = []
        for i, lik in enumerate(self.likelihood_arr):
            var_arr.append(
                lik.conditional_var(f[:, i][:, None])
            )

        return np.array(var_arr)

    def conditional_mean(self, f):
        mu_arr = []
        for i, lik in enumerate(self.likelihood_arr):
            mu_arr.append(
                lik.conditional_mean(f[:, i][:, None])
            )

        return np.array(mu_arr)

    def conditional_samples(self, f, generator=None, num_samples=1):
        samples_arr = []
        for i, lik in enumerate(self.likelihood_arr):
            samples_arr.append(
                lik.conditional_samples(
                    f[..., i][..., None],
                    generator=generator,
                    num_samples=num_samples
                )
            )

        return np.array(samples_arr)

    def laplace_approx(self, f):
        laplace_approx_arr = []
        for i, lik in enumerate(self.likelihood_arr):
            laplace_approx_arr.append(
                lik.laplace_approx(f[:, i][:, None])
            )

        return np.array(laplace_approx_arr)


class GaussianProductLikelihood(ProductLikelihood):

    @property
    def variance(self):
        output_dim = len(self.likelihood_arr)
        var_arr = batch_or_loop(
            lambda lik:  lik.variance,
            [self.likelihood_arr],
            [0],
            dim = output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.likelihood_arr)
        )

        return var_arr


class BlockGaussianProductLikelihood(ProductLikelihood):

    @property
    def block_size(self):
        # TODO: assuming that all product likelihoods 
        return self.likelihood_arr[0].block_size

    @property
    def variance(self):
        output_dim = len(self.likelihood_arr)
        var_arr = batch_or_loop(
            lambda lik:  lik.variance,
            [self.likelihood_arr],
            [0],
            dim = output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.likelihood_arr)
        )

        return np.vstack(var_arr)

    @property
    def precision(self):
        output_dim = len(self.likelihood_arr)
        precision_arr = batch_or_loop(
            lambda lik:  lik.precision,
            [self.likelihood_arr],
            [0],
            dim = output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.likelihood_arr)
        )

        return np.vstack(precision_arr)

