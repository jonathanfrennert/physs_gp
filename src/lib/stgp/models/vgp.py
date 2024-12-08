import objax
import jax
import jax.numpy as np
import numpy as onp
from typing import Optional, Tuple
import chex

from .. import settings
from ..decorators import strict_mode_check, ensure_data
from ..dispatch import dispatch
from ..dispatch import evoke

from ..core import Model, Posterior
from . import GP, BatchGP
from ..kernels import Kernel
from ..inference import Variational
from ..transforms import Transform
from ..computation.log_marginal_likelihoods import *
from ..likelihood import get_product_likelihood
from ..kernels import RBF
from ..approximate_posteriors import MeanFieldApproximatePosterior
from ..sparsity import NoSparsity
from ..utils.utils import ensure_module_list, fix_prediction_shapes
from ..defaults import get_default_kernel, get_default_likelihood, get_default_independent_prior

@dispatch(Model, 'Variational')
class VGP(Posterior):
    def __init__(
        self, 
        X=None, 
        Y=None, 
        Z = None,
        data = None,
        inference: 'Variational'=None, 
        approximate_posterior: 'Posterior'=None, 
        likelihood: 'Likelihood'=None, 
        kernel: 'Kernel'=None, 
        prior: 'Transform'=None, 
        whiten=False, 
        minibatch_size=None,
        ell_samples=None,
        prediction_samples=None,
        **kwargs
    ):

        # will save X, Y as a property
        super(VGP, self).__init__(X, Y, data, **kwargs)

        self.inference = inference
        self._likelihood = likelihood
        self._prior = prior
        self.kernel = kernel
        self.approximate_posterior = approximate_posterior
        self.whiten = whiten
        self.minibatch_size = minibatch_size
        self._Z = Z  
        self.ell_samples = ell_samples
        self.prediction_samples = prediction_samples

        self.set_defaults()
        self.fix_inputs()

    @property
    def X(self):
        return self.data.X 

    def log_marginal_likelihood(self, X=None, Y=None):
        raise NotImplementedError()

    @property
    def output_dim(self): return self.data.Y.shape[1]

    @property
    def input_dim(self): return self.output_dim

    @property
    def input_space_dim(self): return self.data.X.shape[1]

    @property
    def likelihood(self): return self._likelihood

    @property
    def prior(self): return self._prior

    def fix_inputs(self):
        """ Convert all inputs into a consistent format """

        if type(self.likelihood) == list:
            self._likelihood = get_product_likelihood(self._likelihood)
        elif type(self.likelihood).__name__ == 'Gaussian':
            # Internally we only deal with product likelihoods and independent objects
            self._likelihood = get_product_likelihood([self._likelihood])


    def set_defaults(self):
        # Figure out which prior mode is being used (kernel vs prior)

        if (self.kernel is not None) and (self.prior is not None):
            raise RuntimeError('Only kernel or a prior must be passed')

        if self.prior is None:
            # construct an independent prior for each latent function

            # Only set a default kernel if we are in kernel mode and one has not been passed
            if self.kernel is None:
                self.kernel = get_default_kernel(self.input_space_dim, self.input_dim)
            else:
                if type(self.kernel) is not list:
                    self.kernel = [self.kernel]

            # Construct independent prior

            X_ref = self.data._X
            sparsity = [
                NoSparsity(Z_ref = X_ref) 
                for q in range(self.output_dim)
            ]

            self._prior = get_default_independent_prior(
                sparsity,
                self.input_space_dim, 
                self.input_dim, 
                kernel_list=self.kernel,
                Z = self._Z,
            )

        if self.inference == None:
            self.inference = Variational(
                whiten=self.whiten,
                minibatch_size=self.minibatch_size,
                ell_samples = self.ell_samples,
                prediction_samples = self.prediction_samples
            )

        if self.likelihood == None:
            # Default Gaussian liklelihood
            self._likelihood = get_default_likelihood(self.output_dim)


        if self.approximate_posterior is None:
            # Assume independent latents and that they have the same dimension
            base_prior = self.prior.base_prior

            self.approximate_posterior = MeanFieldApproximatePosterior(
                dim_list=[base_prior.get_sparsity_list()[0].Z.shape[0]]*base_prior.output_dim
            )

    def get_objective(self):

        elbo = self.inference.ELBO(
            self.data,
            self.likelihood,
            self.prior,
            self.approximate_posterior
        )

        return -elbo

    def mean(self, XS):
        return self.mean_blocks(XS)


    def covar(self, X1, X2):
        return self.covar_blocks(X1, X2)

    def mean_blocks(self, XS):
        mu, _ = self.predict_f(XS, diagonal=True, squeeze=False, fix_shapes=False)
        mu = np.reshape(mu, [self.output_dim, XS.shape[0], 1])

        chex.assert_shape(mu, [self.output_dim, XS.shape[0], 1])
        return mu

    def var(self, XS):
        _, var = self.predict_f(XS, diagonal=True, squeeze=False)

        var = np.reshape(var, [self.output_dim, XS.shape[0], 1])
        return var

    def covar_blocks(self, XS_1, XS_2, X=None, Y=None):
        var_arr =  self.inference.predictive_covar(
            XS_1, 
            XS_2, 
            self.data, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior
        )

        chex.assert_shape(var_arr, [1, self.output_dim, XS_1.shape[0], XS_2.shape[0]])
        return var_arr[0]

    def var_blocks(self, XS, X=None, Y=None):
        return self.var(XS)

    def predict_latents(self, XS, diagonal=True, squeeze=True):
        mean, var = self.inference.predict_latents(
            XS, 
            self.data, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior,
            diagonal=diagonal
        )

        # TODO: make consistent with predict_f/y
        # TODO: make work is diagonal=False
        # ensure shap is [N, P]
        P = mean.shape[-1]
        N = XS.shape[0]

        if False:
            mean = np.reshape(mean, [N, P])
            var = np.reshape(var, [N, P])

        if squeeze:
            return np.squeeze(mean), np.squeeze(var)

        return mean, var

    def predict_f(self, XS, diagonal=True, squeeze=False, output_first = False, fix_shapes = False, num_samples = None, posterior=False):

        mean, var = self.inference.predict_f(
            XS, 
            self.data, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior,
            diagonal=diagonal,
            num_samples = num_samples,
            posterior=posterior
        )
        if fix_shapes:
            chex.assert_rank([mean, var], [3, 4])

            mean, var = fix_prediction_shapes(
                mean, 
                var,
                diagonal = diagonal,
                squeeze = squeeze,
                output_first = output_first
            )

        return mean, var

    def predict_y(self, XS, diagonal=True, squeeze=True, output_first = False, num_samples = None, posterior = False, fix_shapes=True):
        P = self.prior.output_dim
        N = XS.shape[0]

        mu_arr, var_arr =  self.inference.predict_y(
            XS, 
            self.data, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior,
            diagonal=diagonal,
            num_samples = num_samples,
            posterior=posterior
        )

        chex.assert_rank([mu_arr, var_arr], [3, 4])

        if fix_shapes:
            mu_arr, var_arr = fix_prediction_shapes(
                mu_arr, 
                var_arr,
                diagonal = diagonal,
                squeeze = squeeze,
                output_first = output_first
            )


        return mu_arr, var_arr

    def natural_gradient_update(self, learning_rate, enforce_psd_type=None, prediction_samples=None):
        natgrad_fn = evoke('natural_gradients', self, self.approximate_posterior)

        return natgrad_fn(
            self,
            learning_rate,
            enforce_psd_type,
            prediction_samples
        )

    def samples(self, XS, diagonal=True, num_samples = None, posterior = False):
        samples_arr =  self.inference.samples(
            XS, 
            self.data, 
            self.likelihood, 
            self.prior,
            self.approximate_posterior,
            diagonal=diagonal,
            num_samples = num_samples,
            posterior=posterior
        )

        # [samples, N, P, B]
        chex.assert_rank(samples_arr, 4)

        return samples_arr

    def natural_gradient(self, learning_rate):
        raise NotImplementedError()

    def confidence_intervals(self, XS, num_samples=None, **kwargs):
        """ Returns the median and the 95% confidence intervals. """
        return evoke('confidence_intervals', self)(XS, self, num_samples = num_samples, **kwargs)

    def nlpd(self, XS, YS, num_samples=None):
        return evoke('nlpd', self)(XS, YS, self, num_samples=num_samples)

    def fix(self):
        """ Hold all parameters.  """
        self.likelihood.fix()
        self.prior.fix()
        self.approximate_posterior.fix()

    def release(self):
        """ Un-hold all parameters.  """
        self.likelihood.release()
        self.prior.release()
        self.approximate_posterior.release()
