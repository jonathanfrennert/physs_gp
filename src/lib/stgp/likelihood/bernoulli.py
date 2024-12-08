"""Bernoulli likelihood."""
import objax
import chex
import jax
import jax.numpy as np
from . import DiagonalLikelihood
from ..computation.parameter_transforms import inv_positive_transform, positive_transform, inv_probit
from ..computation.general import log_bernoulli


class Bernoulli(DiagonalLikelihood):
    """Bernoulli likelihood."""

    def __init__(self, link_fn = inv_probit):
        self.link_fn = link_fn

    def log_likelihood_scalar(self, y, f):
        ll = log_bernoulli(y, self.link_fn(f))
        return ll

    def conditional_mean(self, f):
        return self.link_fn(f)

    def conditional_var(self, f):
        p = self.conditional_mean(f)
        return p * (1 - p)


    def conditional_samples(self, f, num_samples = 1, generator=None):
        chex.assert_rank(f, 2)

        uniform_samples = objax.random.uniform(
            [num_samples, f.shape[0], f.shape[1]],
            generator = generator
        )

        samples = jax.vmap(lambda s: np.where(s < f, np.zeros_like(f), np.ones_like(f)))(uniform_samples)

        samples = samples[..., None]
        # [num_samples, f.shape]
        chex.assert_rank(samples, 4)
        return samples

