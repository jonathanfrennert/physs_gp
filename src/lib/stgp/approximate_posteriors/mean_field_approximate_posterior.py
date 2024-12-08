import jax
import jax.numpy as np
import objax
import chex
from .. import settings
from ..utils.utils import ensure_module_list, get_batch_type
from batchjax import batch_or_loop
from ..dispatch import _ensure_str

from ..transforms  import LinearTransform, NonLinearTransform
from . import ApproximatePosterior, GaussianApproximatePosterior
from ..computation.matrix_ops import vectorized_lower_triangular_cholesky, lower_triangle
import chex

from typing import Optional, List

def batch_over_posteriors(post_list, fn):
    arr = batch_or_loop(
        fn,
        [post_list],
        [0],
        dim = len(post_list),
        out_dim = 1,
        batch_type = get_batch_type(post_list)
    )
    return arr

class MeanFieldApproximatePosterior(ApproximatePosterior):
    def __init__(self, dim_list: List[int]=None, approximate_posteriors: Optional[List[GaussianApproximatePosterior]]=None):
        super(MeanFieldApproximatePosterior, self).__init__()

        if approximate_posteriors is None:
            self.approx_posteriors = objax.ModuleList([
                GaussianApproximatePosterior(dim=dim_list[q])
                for q in range(len(dim_list))
            ])

        elif type(approximate_posteriors) is list: 
            self.approx_posteriors = objax.ModuleList(approximate_posteriors)
        else:
            self.approx_posteriors = approximate_posteriors

        self.num_of_latents = len(self.approx_posteriors)

        if _ensure_str(self.approx_posteriors[0]) == 'MeanFieldAcrossDataApproximatePosterior':
            self.meanfield_over_data = True
        else:
            self.meanfield_over_data = False


        

    def get_variational_params(self):
        return self.m, self.S_chol

    @property
    def m(self):
        m_arr = batch_over_posteriors(
            self.approx_posteriors,
            lambda latent:  latent.m
        )

        return m_arr

    @property
    def S_chol(self):
        arr = batch_over_posteriors(
            self.approx_posteriors,
            lambda latent:  latent.S_chol
        )

        return arr

    @property
    def S(self):
        arr = batch_over_posteriors(
            self.approx_posteriors,
            lambda latent:  latent.S
        )
        return arr

    @property
    def S_diag(self):
        arr = batch_over_posteriors(
            self.approx_posteriors,
            lambda latent:  latent.S_diag
        )
        return arr

    def fix(self):
        for q in self.approx_posteriors:
            q.fix()

    def release(self):
        for q in self.approx_posteriors:
            q.release()
