""" Power likelihoods of the form p(y|f)^a """
import jax.numpy as np
from . import Likelihood, FullLikelihood, DiagonalLikelihood, BlockDiagonalLikelihood

from .. import Parameter

class PowerLikelihood(Likelihood):
    def __init__(self, base_lik = None, a = None, train=True):

        if base_lik is None:
            raise RuntimeError('Likelihood must be passed.')


        self.parent = base_lik

        if a is None: 
            a = 1.0

        a = np.array(a)

        self.a_param = Parameter(
            a, 
            constraint = 'positive', 
            name ='PowerLikelihood/a', 
            train=train
        )

    @property
    def a(self):
        return self.a_param.value


    def log_likelihood_scalar(self, y, f):
        parent_ll = self.parent.log_likelihood_scalar(y, f)

        return parent_ll * self.a
