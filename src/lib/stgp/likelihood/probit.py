"""Probit likelihood."""
import objax
import jax.numpy as np
from . import DiagonalLikelihood
from ..computation.parameter_transforms import inv_positive_transform, positive_transform, inv_probit
from ..computation.general import log_inv_probit


class Probit(DiagonalLikelihood):
    """Probit likelihood."""

    def __init__(self,  nu = 1e-6):
        self.nu = nu

    def log_likelihood_scalar(self, y, f):
        scaled_f = f * (1/self.nu)
        log_ll = log_inv_probit(scaled_f)
        return log_ll

