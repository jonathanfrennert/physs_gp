"""Single input and outputs transforms."""
from .transform import Transform, LinearTransform, ElementWiseTransform, LatentSpecific

import jax.numpy as np
import objax
from ..utils.utils import ensure_module_list
from ..computation.parameter_transforms import softplus, inv_softplus, inv_probit
from ..parameter import Parameter

class CompositeTransform(Transform):
    def __init__(self, transform_arr):
        self.transform_arr = transform_arr

    def forward(self, x):
        res = x
        for t in self.transform_arr:
            res = t.forward(res)
        return res

    def inverse(self, x):
        res = x
        for t in self.transform_arr[::-1]:
            res = t.inverse(res)
        return res

class InputMeanFunction(LinearTransform, LatentSpecific):
    def __init__(self, latent):
        self._parent = latent

    def mean(self, X):
        return X[:, 0][:, None]


class Identity(ElementWiseTransform):
    def forward(self, x):
        return x

    def inverse(self, f):
        return f

class ReverseFlow(ElementWiseTransform):
    def __init__(self, base_flow):
        self.base_flow = base_flow

    def forward(self, x):
        return self.base_flow.inverse(x)

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        return self.base_flow.forward(f)


class Square(ElementWiseTransform):
    """Expontial Function."""

    def forward(self, x):
        """Compute f=T(x)."""
        return x**2

    def inverse(self, f):
        raise RuntimeError()

class Exp(ElementWiseTransform):
    """Expontial Function."""

    def forward(self, x):
        """Compute f=T(x)."""
        return np.exp(x)

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        return np.log(f)


class Log(Exp):
    """Log function. Inverse of Exp function."""

    def forward(self, x):
        """Compute f=T(x)."""
        return super(Log, self).inverse(x)

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        return super(Log, self).forward(f)

class Softminus(ElementWiseTransform):

    def forward(self, x):
        """Compute f=T(x)."""
        return inv_softplus(x)

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        return softplus(f)

class Softplus(ElementWiseTransform):

    def forward(self, x):
        """Compute f=T(x)."""
        return softplus(x)

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        return inv_softplus(f)

class Affine(ElementWiseTransform):
    """Affine Function."""
    def __init__(self, a, b, train=True):

        # a is std, b is mean
        self.a_param = Parameter(
            np.array(a), 
            constraint=None, 
            name ='Affine/a', 
            train=train
        )

        self.b_param = Parameter(
            np.array(b), 
            constraint=None, 
            name ='Affine/b', 
            train=train
        )

    @property
    def a(self):
        return self.a_param.value

    @property
    def b(self):
        return self.b_param.value

    def forward(self, x):
        return x * self.a + self.b

    def inverse(self, f):
        return (f - self.b) / self.a


class BoxCox(ElementWiseTransform):
    """
    Boxcox Function

        fk = (f0^lam - 1)/lam

    where lam > 0.
    """

    def __init__(self, lam, train=True):

        self.lam_param = Parameter(
            np.array(lam), 
            constraint='positive', 
            name ='BoxCox/lam', 
            train=train
        )

    def forward(self, x):
        lam = self.lam_param.value
        return (np.power(x, lam) - 1)/lam

    def inverse(self, f):
        lam = self.lam_param.value
        return np.power((f*lam)+1, 1./lam )


class Sinh_Arcsinh(ElementWiseTransform):
    """Sinh_Arcsinh Function."""


class Tanh(ElementWiseTransform):
    """Sinh_Arcsinh Function."""

class InvProbit(ElementWiseTransform):
    """Sinh_Arcsinh Function."""
    def forward(self, x):
        return inv_probit(x)

