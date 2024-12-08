import jax.numpy as np
import objax

from . import GaussianApproximatePosterior

class MM_GaussianInnerLayerApproximatePosterior(GaussianApproximatePosterior):
    def __init__(self, kernel, *args, **kwargs):
        super(MM_GaussianInnerLayerApproximatePosterior, self).__init__(*args, **kwargs)

        self.kernel = kernel
