import objax
from .models import GP
from .kernels import RBF
from .likelihood import Gaussian, get_product_likelihood
from .transforms import Independent
from typing import List, Optional
from .sparsity import NoSparsity, FullSparsity
import warnings

def get_default_kernel(input_dim: int, num_latents: int) -> List['Kernel']:
    warnings.warn('Using default RBF kernel')
    return [
        RBF(
            lengthscales=[1.0 for d in range(input_dim)],
            input_dim=input_dim
        )
        for q in range(num_latents)
    ]

def get_default_likelihood(num_outputs: int) -> List['Likelihood']:
    warnings.warn('Using default Product Gaussian Likelihood')
    return get_product_likelihood([
        Gaussian(variance=1.0)
        for p in range(num_outputs)
    ])

def get_default_independent_prior(sparsity, input_dim: int, num_latents: int, kernel_list: Optional['Kernel'] = None, Z: Optional['np.ndarray'] = None) -> 'Prior':
    warnings.warn('Using default Independent prior')

    if kernel_list is None:
        kernel_list = get_default_kernel(input_dim, num_latents)

    return Independent(
        latents = [
            GP(
                kernel = kernel_list[q],
                sparsity = sparsity[q]
            )
            for q in range(num_latents)
        ],
        prior=True
    )
