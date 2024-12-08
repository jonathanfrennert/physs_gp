import jax.numpy as np
import objax
from ..core import Posterior
from ..dispatch import evoke
from .. import settings
import chex

class ApproximatePosterior(Posterior):
    def __init__(self):
        super(ApproximatePosterior, self).__init__()

        self.generator = objax.random.Generator(seed=0)

    def fix(self):
        """ Hold all variational params for training """
        raise NotImplementedError()
