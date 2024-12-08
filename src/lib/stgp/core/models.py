"""
Defines the abstract model classes.
Models are split into priors (no Y) and posteriors (has Y)
"""
import objax
import jax.numpy as np
import numpy as onp
from typing import Optional, Tuple
from ..utils import utils
from ..data import Data

class Model(objax.Module):
    """
    Base model class for almost all objects
    """
    def __init__(self, **kwargs):
        super(Model, self).__init__()

    def set_defaults(self):
        raise NotImplementedError()

    def fix_inputs(self):
        raise NotImplementedError()

    @property
    def name(self) -> int:
        return 'model_checkpoint'

    @property
    def input_dim(self) -> int:
        """ If a transformation the number of inputs it transforms. """
        raise NotImplementedError()

    @property
    def output_dim(self) -> int:
        """ Number of outputs """
        raise NotImplementedError()

    def mean(self, XS: np.ndarray) -> np.ndarray :
        """ Rank 2 mean (output_dim * N x 1) """
        raise NotImplementedError()

    def b_mean(self, XS: np.ndarray) -> np.ndarray :
        """ Rank 2 mean (output_dim * N x 1) """
        raise NotImplementedError()

    def mean_blocks(self, XS: np.ndarray) -> np.ndarray :
        """ Rank 3 mean (output_dim x N x 1) """
        raise NotImplementedError()

    def b_mean_blocks(self, XS: np.ndarray) -> np.ndarray :
        """ Rank 3 mean (output_dim x N x 1) whilst batching over XS """
        raise NotImplementedError()

    def var(self, XS: np.ndarray) -> np.ndarray :
        """ Rank 2 diagonal variance (output_dim * N x 1) """
        raise NotImplementedError()

    def var_blocks(self, XS: np.ndarray) -> np.ndarray :
        """ Rank 3 diagonal variance (output_dim x N x 1) """
        raise NotImplementedError()

    def full_var(self, XS: np.ndarray) -> np.ndarray :
        """ Rank 2 full variance (output_dim * N x output_dim * N) """
        raise NotImplementedError()

    def full_var_blocks(self, XS: np.ndarray) -> np.ndarray :
        """ Rank 3  variance (output_dim x N x N) """
        raise NotImplementedError()

    def covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray :
        """ Rank 2 full covariance (output_dim * N1 x output_dim * N2) """
        raise NotImplementedError()

    def b_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray :
        """ Rank 2 full covariance (output_dim * N1 x output_dim * N2) whilst batching over X1 and X2"""
        raise NotImplementedError()

    def covar_blocks(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray :
        """ Rank 3  covariance (output_dim x N1 x N2) """
        raise NotImplementedError()

    def b_covar_blocks(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray :
        """ Rank 3  covariance (output_dim x N1 x N2) whilst batching over X1 and X2"""
        raise NotImplementedError()

    def sample(self, X1):
        raise NotImplementedError()

    def state_space_representation(self, X_s):
        raise NotImplementedError()

    def print(self):
        #dicts should be the same ordering
        param_dict = utils.get_parameters(self)

        fixed_params = utils.get_fixed_params(self, replace_name=True)

        for k, v in param_dict.items():
            if k in fixed_params:
                print(f'(fixed) {k}: {v}')
            else:
                print(f'{k}: {v}')

    def print_fixed_params(self):
        print(utils.get_fixed_params(self))

    def checkpoint(self, name=None):
        if name is None:
            name = self.name

        objax.io.save_var_collection(f'{name}.npz', self.vars())

    def load_from_checkpoint(self, name=None):
        if name is None:
            name = self.name

        objax.io.load_var_collection(f'{name}.npz', self.vars())

    def get_fixed_params(self):
        return utils.get_fixed_params(self)

class Prior(Model):
    def get_sparsity_list(self):
        """ Return list of sparsity objects """
        raise NotImplementedError()

    def get_Z(self):
        """ Return rank 2 inducing locations (output_dim * N x D) """
        raise NotImplementedError()

    def get_Z_blocks(self):
        """ Return  inducing locations (output_dim x N x D) """
        raise NotImplementedError()

    def get_Z_stacked(self):
        """ 
        Return stacked inducing locations (Q x output_dim x N x D).

        This required when batching over independent priors.
        """
        raise NotImplementedError()

class Posterior(Model):
    def __init__(self, X=None, Y=None, data=None, latent_y=False, latent_x=False, **kwargs):
        if X is not None and Y is not None:
            if data is None:
                data = Data(X, Y)

        self.data = data

    @property
    def X(self):
        raise NotImplementedError()

    @property
    def Y(self):
        raise NotImplementedError()

    def log_marginal_likelihood(self, X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None):
        raise NotImplementedError()

    @property
    def prior(self):
        raise NotImplementedError()

    @property
    def likelihood(self):
        raise NotImplementedError()

    def predict_f(self, XS: np.ndarray, *args, **kwargs):
        """ 
        Predict p(f^* | data) 

        Should support kwargs: 
            diagonal: bool
            squeeze: bool

        Typical Output Shapes:
            diagonal == True:
                mean: N x P
                var: N x P

            diagonal == False:
                mean: N x P x 1
                var: N x P x P

        Optional kwargs:
            output_first:
                diagonal == True:
                    mean: P x N
                    var: P x N
                diagonal == False:
                    n/a

        """
        raise NotImplementedError()

    def predict_y(self, XS: np.ndarray, *args, **kwargs):
        """ Predict p(y^* | data). For output shapes see <predict_f>."""
        raise NotImplementedError()

    def posterior_blocks(self, *args, **kwargs):
        """ Return the mean and variance of the posterior in blocks, as determined by the likelihood."""
        raise NotImplementedError()

    def posterior(self, *args, **kwargs):
        """ Return the mean and variance of the posterior"""
        raise NotImplementedError()

    def confidence_intervals(self, XS):
        """ Returns the median and the 95% confidence intervals. """
        raise NotImplementedError()

    def npld(self, XS, YS):
        """ Returns the negative log predictive likelihood """
        raise NotImplementedError()


        



