"""(Probably) Hetereogenous Gaussian likelihood."""
import objax
import jax.numpy as np
from . import DiagonalLikelihood
from ..computation.gaussian import log_gaussian_scalar
from ..computation.matrix_ops import hessian


class HetGaussian(DiagonalLikelihood):
    def __init__(self, link_fn=None):
        # hack for now, when using a mean-field it assumed that there is one likelihood
        #  per output
        self.likelihood_arr = [self]
        if link_fn is None:
            link_fn = lambda x: x**2
        else:
            link_fn = np.exp
        self.link_fn = link_fn

    def log_likelihood_scalar(self, y, f):
        return log_gaussian_scalar(y, f[0], self.link_fn(f[1]))

    def conditional_var(self, f):
        return np.array([
            [0, 0],
            [0, np.squeeze(self.link_fn(f[1]))]
        ])

    def conditional_mean(self, f):
        return f[0]

    def log_hessian_scalar(self, y, f):
        hess =  hessian(lambda t: self.log_likelihood_scalar(y, t), 0)(f)

        return np.diag(np.diag(hess))

    def laplace_approx(self, f):
        return np.array([
            [0, 0],
            [0, -1/np.squeeze(self.link_fn(f[1]))]
        ])



