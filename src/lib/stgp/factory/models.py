import numpy as np

from ..model import GP
from ..likelihood import Gaussian
from ..kernel import RBF

def get_gpr(X: np.ndarray, Y: np.ndarray) -> GP:

    m = GP(
        X, 
        Y,
        likelihood=Gaussian(variance=1.0),
        kernel=RBF(variance=1.0, lengthscales=np.array([1.0]))
    )

    return m
