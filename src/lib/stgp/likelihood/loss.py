import jax.numpy as np
from .likelihood import DiagonalLikelihood
from ..computation.parameter_transforms import  positive_transform

class LossLikelihood(DiagonalLikelihood):
    pass

class NonZeroLoss(LossLikelihood):
    def __init__(self,  scale = 1.0, nu=1.0):
        self.scale = scale
        self.nu = nu

    def fix(self):
        pass

    def release(self):
        pass


    def likelihood_scalar(self, y, f):
        # f^2 so that both negative and positive values are treated the same
        #return self.scale * 1/(self.nu*(f**2) + 1e-8)
        return  self.scale*np.tanh(positive_transform(self.nu*(f**2) + 1e-8))

    def log_likelihood_scalar(self, y, f):
        return np.log(self.likelihood_scalar(y, f))

    def conditional_mean(self, f):
        # for debugging and plotting  since y is not required
        return self.likelihood_scalar(f, f)
