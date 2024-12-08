""" Functional diff op kernel functions so we can jit them """

import jax
import jax.numpy as np
from jax import jit, grad, jacfwd, jacrev
from functools import partial
import chex

from ..computation.matrix_ops import hessian

#@partial(jit, static_argnums=(2, 3, 4))
def FirstOrderDerivativeKernel_compute_derivatives(x1, x2, var_fn, input_index, d_computed):
    """
    Let T to denote a differential operator: d/dt

    Let D = 
        I., .(T), 
        (T).,    (T).(T)        

    be a matrix of linear operators (where . denotes the operator input, and we ignore transposes). 

    Then The full joint kernel is given by (ignoring transposes, and abusing the kronecker product notation):

        K ⊗ D

    When K is scalar this is given as

        K,       K(T)             
        (T)K,    (T)K(T)        

    This means that the output is ordered by [f_1, (T)f_1, ..., f_B, (T)f_B]^T.
    """

    #B x B
    k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])

    # compute blocks

    # variable name notation
    # res<x1 diff_order><x2 diff order>

    # Computes
    # [K]
    #B x B
    res00 = k(x1, x2)

    B = res00.shape[0]

    # Computes
    # [(T)K]
    # B x B x D
    res10 = jacfwd(k, argnums=(0))(x1, x2)

    # Computes
    # K(T)
    # B x B x D
    res01 = jacfwd(k, argnums=(1))(x1, x2)


    # Computes
    # (T)K(T)
    #  B x B x D x D
    res11 = jacfwd(jacfwd(k, argnums=(0)), argnums=(1))(x1, x2)

    # Construct full matrix
    # K,       K(T)
    # (T)K,    (T)K(T)


    # for a given B_i, B_j compute the derivate kernels
    #def get_K(i, j):
    #    return np.array([
    #        [res00[i, j],       res01[i, j, self.input_index]], # f
    #        [res10[i, j, self.input_index],    res11[i, j, self.input_index, self.input_index]], # df/dt
    #    ])

    def get_K(i, j):
        # use transpose to force symmetric
        return np.array([
            [res00[i, j],       res01[i, j, input_index].T], # f
            [res10[i, j, input_index],    res11[i, j, input_index, input_index]], # df/dt
        ])

    # forces B - D format
    # stack all derivate kernels over each BxB element 
    K = np.block([
        [
            get_K(b1, b2) 
            for b2 in range(B) 
        ]
        for b1 in range(B) 
    ])

    chex.assert_rank(K, 2)
    chex.assert_shape(K, [B*d_computed, B*d_computed])

    return K



# TODO: why cant i jit here?
#@partial(jit, static_argnums=(2, 3, 4))
def SecondOrderOnlyDerivativeKernel_compute_derivatives(x1, x2, var_fn, input_index, d_computed):

    """
    Let T to denote a differential operator: d/dt

    Let D = 
        I.,  .(T^2), 
        (T)^2.,  (T)^2.(T^2) 

    be a matrix of linear operators (where . denotes the operator input, and we ignore transposes). 

    Then The full joint kernel is given by (ignoring transposes, and abusing the kronecker product notation):

        K ⊗ D

    When K is scalar this is given as

        K,        K(T^2)             
        (T)^2K,   (T)^2K(T^2)   

    This means that the output is ordered by [f_1, (T^2)f_1, ..., f_B, (T^2)f_B]^T.
    """

    #B x B
    k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])

    # compute blocks

    # variable name notation
    # res<x1 diff_order><x2 diff order>

    # Computes
    # [K]
    #B x B
    res00 = k(x1, x2)

    B = res00.shape[0]

    # Computes
    # (T^2)K
    #  B x B x D x D
    res20 = hessian(k, argnums=(0))(x1, x2)

    # Computes
    # K(T^2)
    #  B x B x D x D
    res02 = hessian(k, argnums=(1))(x1, x2)

    # arg 0 are the first dim, arg1 are the final
    # (T^2)K(T^2)
    #  B x B x D x D x D x D
    res22 = hessian(hessian(k, argnums=(0)), argnums=(1))(x1, x2)

    # Construct full matrix
    # K,       K(T^2)
    # (T)^2K,  (T)^2K(T^2)



    def get_K(i, j):
        return np.array([
            [res00[i, j], res02[i, j, input_index, input_index].T], # f
            [res20[i, j, input_index, input_index], res22[i, j, input_index, input_index, input_index, input_index]], # d^2f/dt^2
        ])

    # stack all derivate kernels over each BxB element 
    K = np.block([
        [
            get_K(b1, b2) 
            for b2 in range(B) 
        ]
        for b1 in range(B) 
    ])

    chex.assert_rank(K, 2)
    chex.assert_shape(K, [B*d_computed, B*d_computed])

    return K

