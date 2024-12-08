from .transform import Transform, LinearTransform, NonLinearTransform, Independent
import jax.numpy as np


class Aggregate(LinearTransform):
    def __init__(self, parent):
        self._parent = parent
         
        # TODO: figure out
        self._input_dim = 1
        self._output_dim = 1

    def forward(self, f):
        raise NotImplementedError()


