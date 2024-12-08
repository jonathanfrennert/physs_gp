""" Useful wrappers """
from ..core import Model, Posterior
from ..dispatch import dispatch
from ..dispatch import evoke

import jax
import objax
import chex

import jax.numpy as np

from ..approximate_posteriors import MeanFieldApproximatePosterior

class LatentPredictor(Model):
    def __init__(self, model):
        self.base_model = model

    @property
    def output_dim(self): return self.base_model.output_dim

    @property
    def input_dim(self): return self.base_model.input_dim

    def mean_blocks(self, XS):
        mu, _ = self.base_model.predict_latents(XS, diagonal=True, squeeze=False)

        mu = np.transpose(mu, [1, 0, 2])

        chex.assert_shape(mu, [self.output_dim, XS.shape[0], 1])
        return mu


    def var_blocks(self, XS):
        _, var = self.base_model.predict_latents(XS, diagonal=True, squeeze=False)
        var = var[..., 0]
        var = np.transpose(var, [1, 0, 2])

        chex.assert_shape(var, [self.output_dim, XS.shape[0], 1])
        return var

    def covar_blocks(self, XS_1, XS_2, X=None, Y=None):
        var_arr =  self.base_model.inference.predictive_latent_covar(
            XS_1, 
            XS_2, 
            self.base_model.data, 
            self.base_model.likelihood, 
            self.base_model.prior,
            self.base_model.approximate_posterior
        )

        var_arr = var_arr[:, 0, ...]

        chex.assert_shape(var_arr, [self.output_dim, XS_1.shape[0], XS_2.shape[0]])
        return var_arr


class MultiObjectiveModel(Model):
    def __init__(self, model_list):
        self.model_list = objax.ModuleList(model_list)
        if False:
            self.approximate_posterior = MeanFieldApproximatePosterior(
                approximate_posteriors = [m.approximate_posterior.approx_posteriors[0] for m in model_list]
            )
        if False:
            self.approximate_posterior = model_list[-1].approximate_posterior

    def get_objective(self):
        obj = 0.0
        for m in self.model_list:
            obj += m.get_objective()

        return obj

    def natural_gradient_update(self, learning_rate, enforce_psd_type=None, prediction_samples=None):
        natgrad_fn = evoke('natural_gradients', self, self.approximate_posterior)

        return natgrad_fn(
            self,
            learning_rate,
            enforce_psd_type,
            prediction_samples
        )

