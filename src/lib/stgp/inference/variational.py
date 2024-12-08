"""Variational inference class."""
from . import Inference
from .. import settings
from ..dispatch import evoke

from ..computation.elbos import *

import jax
import jax.numpy as np
import objax
import chex

class Variational(Inference):
    """Variational inference class."""
    def __init__(self, whiten=False, minibatch_size=False, ell_samples=1, prediction_samples=100):
        super(Variational, self).__init__()

        self.whiten = whiten
        self.minibatch_size = minibatch_size
        self.generator = objax.random.Generator(seed=0)
        self.ell_samples=ell_samples
        self.prediction_samples=prediction_samples

    def predict_f(self, XS, data, likelihood, prior, approximate_posterior, diagonal, num_samples=None, posterior=False):

        if data.minibatch:
            # TODO: minibatching only works when sparsity is used. Assert this.
            data.batch()

        mu, var = evoke('predict', likelihood, prior, approximate_posterior, self.whiten)(
            XS, data, likelihood, prior, approximate_posterior, self, self.whiten, diagonal, num_samples = num_samples, posterior=posterior
        )

        return mu, var

    def samples(self, XS, data, likelihood, prior, approximate_posterior, diagonal, num_samples=None, posterior=False):
        if data.minibatch:
            # TODO: minibatching only works when sparsity is used. Assert this.
            data.batch()

        mu = evoke('marginal_prediction_samples', approximate_posterior, likelihood, prior, whiten=self.whiten)(
            XS, data, approximate_posterior, likelihood, prior, self, diagonal, self.whiten, num_samples = num_samples, posterior=posterior
        )


        return mu

    def predict_latents(self, XS, data, likelihood, prior, approximate_posterior, diagonal):
        if data.minibatch:
            # TODO: minibatching only works when sparsity is used. Assert this.
            data.batch()

        return evoke('marginal', 'latents', approximate_posterior, likelihood, prior, whiten=self.whiten)(
            XS, data, approximate_posterior, likelihood, prior, self, 1, self.whiten
        )

    def predict_y(self, XS, data, likelihood, prior, approximate_posterior, diagonal,  num_samples=None, posterior=False):
        pred_mu, pred_var = self.predict_f(XS, data, likelihood, prior, approximate_posterior, diagonal, num_samples = num_samples, posterior=posterior)

        if True:
            pred_y_mu, pred_y_var = evoke('predict_y', approximate_posterior, likelihood, prior)(
                XS, approximate_posterior, likelihood, pred_mu, pred_var, diagonal
            )

            return pred_y_mu, pred_y_var

        return pred_mu, pred_var


    def predictive_covar(self, XS_1, XS_2, data, likelihood, prior, approximate_posterior):
        if data.minibatch:
            # TODO: minibatching only works when sparsity is used. Assert this.
            data.batch()

        sparsity_list = prior.base_prior.get_sparsity_list()

        whiten = self.whiten

        q_m, q_S_chol = evoke('variational_params', approximate_posterior, likelihood, prior.base_prior, whiten)(
            data, approximate_posterior, likelihood, prior, whiten
        )
        q_m = np.array(q_m)
        q_S_chol = np.array(q_S_chol)

        return evoke('marginal_prediction_covar', approximate_posterior, likelihood, prior, sparsity_list[0], whiten=self.whiten)(
            XS_1, XS_2, data, q_m, q_S_chol, approximate_posterior, likelihood, prior, sparsity_list, 1, self.whiten
        )

    def predictive_latent_covar(self, XS_1, XS_2, data, likelihood, prior, approximate_posterior):
        if data.minibatch:
            # TODO: minibatching only works when sparsity is used. Assert this.
            data.batch()

        sparsity_list = prior.base_prior.get_sparsity_list()

        whiten = self.whiten

        q_m, q_S_chol = evoke('variational_params', approximate_posterior, likelihood, prior.base_prior, whiten)(
            data, approximate_posterior, likelihood, prior, whiten
        )
        q_m = np.array(q_m)
        q_S_chol = np.array(q_S_chol)

        latent = prior.base_prior

        return evoke('marginal_prediction_covar', approximate_posterior, likelihood, latent, sparsity_list[0], whiten=self.whiten)(
            XS_1, XS_2, data, q_m, q_S_chol, approximate_posterior, likelihood, latent, sparsity_list, None, self.whiten
        )


    def ELBO(self, data, likelihood, prior, approximate_posterior):
        """
        Args:
            data: Data
            likelihood: Array of P likelihoods
            prior: 
            approximate_posterior:  approximate posterior
        """

        val = evoke('elbo', likelihood, prior, approximate_posterior)(
            data, likelihood, prior, approximate_posterior, self 
        )

        return val
