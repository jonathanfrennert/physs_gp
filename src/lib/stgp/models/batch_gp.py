import objax
import jax
import jax.numpy as np
import numpy as onp
from typing import Optional, Tuple
import chex

from .. import settings
from ..decorators import strict_mode_check, ensure_data
from ..dispatch import dispatch, evoke

from ..core import Model, Posterior
from . import GP
from ..kernels import Kernel
from ..inference import Batch
from ..likelihood import Gaussian, get_product_likelihood, ProductLikelihood, DiagonalLikelihood
from ..kernels import RBF
from ..utils.utils import ensure_module_list
from ..transforms import Independent
from ..defaults import get_default_kernel, get_default_likelihood, get_default_independent_prior

from ..sparsity import NoSparsity

import warnings

@dispatch('Model', 'Batch')
class BatchGP(Posterior):
    def __init__(
        self, 
        X=None, 
        Y=None, 
        data=None, 
        inference: 'Batch'=None, 
        likelihood: 'Likelihood'=None, 
        kernel: 'Kernel'=None, 
        prior: 'Transform' = None, 
        **kwargs
    ):
        # will save X, Y as a property
        super(BatchGP, self).__init__(X, Y, data, **kwargs)

        self.inference = inference
        self._likelihood = likelihood
        self.kernel = kernel
        self._prior = prior

        self.set_defaults()

    @property
    def X(self):
        return self.data.X

    @property
    def Y(self):
        return self.data.Y

    @property
    def likelihood(self): return self._likelihood

    @property
    def prior(self): return self._prior

    @property
    def input_space_dim(self): return self.data.X.shape[1]

    @property
    def output_dim(self): return self.data.Y.shape[1]

    @property
    def input_dim(self): return self.output_dim

    def set_defaults(self):
        """ Replace missing options with defaults """

        # Figure out which prior mode is being used (kernel vs prior)

        if (self.kernel is not None) and (self.prior is not None):
            raise RuntimeError('Only kernel or a prior must be passed')

        if self.prior is None:
            # construct an independent prior for each latent function

            # Only set a default kernel if we are in kernel mode and one has not been passed
            if self.kernel is None:
                self.kernel = get_default_kernel(self.input_space_dim, self.output_dim)
            else:
                if type(self.kernel) is not list:
                    self.kernel = [self.kernel]

            # Pass object to avoid storing multiple copies of X
            X_ref = self.data._X
            sparsity = [
                NoSparsity(Z_ref = X_ref) 
                for q in range(self.output_dim)
            ]

            # Construct independent prior
            self._prior = get_default_independent_prior(
                sparsity,
                self.input_space_dim, 
                self.output_dim, 
                kernel_list=self.kernel
            )

        if self.inference == None:
            self.inference = Batch()

        if self.likelihood == None:
            self._likelihood = get_default_likelihood(self.output_dim)

        else:
            if type(self.likelihood) == list:
                self._likelihood = get_product_likelihood(self._likelihood)
            elif issubclass(type(self.likelihood), DiagonalLikelihood):
                self._likelihood = get_product_likelihood([self.likelihood])


    def log_marginal_likelihood(self):
        nlml = self.inference.neg_log_marginal_likelihood(
            self.data,
            self, 
            self.likelihood,
            self.prior
        )

        chex.assert_rank(nlml, 0)

        return nlml

    def get_objective(self):
        return self.log_marginal_likelihood()

    def mean(self, XS):
        mean_blocks = self.mean_blocks(XS)
        mean = np.reshape(m.mean_blocks(XS), [-1, 1])

        chex.assert_shape(mean, [self.output_dim * XS.shape[0], 1])

        return mean

    def mean_blocks(self, XS):
        mu, _ = self.predict_f(XS, diagonal=True, squeeze=False)

        mu = np.reshape(mu, [self.output_dim, XS.shape[0], 1])

        chex.assert_shape(mu, [self.output_dim, XS.shape[0], 1])

        return mu

    def var_blocks(self, XS):
        _, var = self.predict_f(XS, diagonal=True, squeeze=False)

        var = np.reshape(var, [self.output_dim, XS.shape[0], 1])
        chex.assert_shape(var, [self.output_dim, XS.shape[0], 1])

        return var

    def covar(self, XS_1, XS_2):
        assert self.output_dim == 1
        covar =  self.covar_blocks(XS_1, XS_2)[0]


        chex.assert_shape(covar, [XS_1.shape[0], XS_2.shape[0]])
        return covar

    def covar_blocks(self, XS_1, XS_2):
        var_arr =  self.inference.predictive_covar(
            XS_1, XS_2, self.data, self, self.likelihood, self.prior
        )

        chex.assert_shape(var_arr, [self.output_dim, XS_1.shape[0], XS_2.shape[0]])

        return var_arr


    def predict_f(self, XS,  diagonal=True, squeeze=False):
        mu_arr, var_arr =  self.inference.predict_f(
            XS, self.data, self, self.likelihood, self.prior, diagonal=diagonal
        )

        if squeeze:
            mu_arr, var_arr = np.squeeze(mu_arr), np.squeeze(var_arr) 

        return mu_arr, var_arr

    def predict_y(self, XS, diagonal=True, squeeze=False):
        mu_arr, var_arr =  self.inference.predict_y(
            XS, self.data, self, self.likelihood, self.prior, diagonal=diagonal
        )

        if squeeze:
            mu_arr, var_arr = np.squeeze(mu_arr), np.squeeze(var_arr) 

        return mu_arr, var_arr

    def confidence_intervals(self, XS):
        """ Returns the median and the 95% confidence intervals. """
        return evoke('confidence_intervals', self)(XS, self)

    def predict_blocks(self, XS, group_size: int, block_size: int, diagonal=True, squeeze=False):
        """ Returns predictions with the same block size as the likelihood. """ 
        X, Y = self.X, self.Y

        mu_arr, var_arr =  self.inference.predict_f_blocks(
            XS, 
            self.data,
            self, 
            self.likelihood, 
            self.prior, 
            group_size, 
            block_size
        )

        if squeeze:
            mu_arr, var_arr = np.squeeze(mu_arr), np.squeeze(var_arr) 

        return mu_arr, var_arr

    def posterior_blocks(self, return_lml=False):
        mu, var =  self.predict_blocks(
            self.data.X, 
            group_size=1, 
            block_size=self.likelihood.block_size,
            diagonal=False
        )


        if return_lml:
            lml = -self.get_objective()
            return lml, mu, var

        return mu, var

    def posterior(self, diagonal=True):
        return self.predict_f(self.data.X, diagonal=diagonal)

    def nlpd(self, XS, YS, num_samples=None):
        return evoke('nlpd', self)(XS, YS, self, num_samples=num_samples)

    def fix(self):
        self.likelihood.fix()
        self.prior.fix()

    def release(self):
        self.likelihood.release()
        self.prior.release()
