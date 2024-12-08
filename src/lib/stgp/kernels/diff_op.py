from . import Kernel
import jax
import jax.numpy as np
from jax import jacfwd, jacrev, grad
from ..computation.matrix_ops import hessian
from .white_noise import WhiteNoise

import chex

from .diff_op_utils import FirstOrderDerivativeKernel_compute_derivatives, SecondOrderOnlyDerivativeKernel_compute_derivatives

class DerivativeKernel(Kernel):
    """
    Accecpts a parent kernel OR a parent model
    """
    def __init__(self, parent_kernel = None) :
        self.parent_kernel = parent_kernel
        self.active_dims = None
        self.parent = self.parent_kernel

    def K_diag(self, X):
        chex.assert_rank(X, 1)
        return jax.vmap(lambda x: self.K(x[..., None], x[..., None]))(X[..., None])
        

    def _K(self, X1, X2):
        """ This is is being used as a prior kernel therefore we can pass through the
        kernel function """

        return self._K_from_fn(X1, X2, self.parent_kernel.K)

    def K_from_fn(self, X1, X2, var_fn):
        return self._K_from_fn(X1, X2, var_fn)

    def to_ss(self, X_spatial=None):
        """
        The state-space representation using the Kalman computes the derivates implicitely. Therefore we just pass to the base, (non-derivatie) kernel. 
        """
        return self.base.to_ss(X_spatial=X_spatial)


    def state_space_dim(self):
        # only need return the time dim
        return self.base.state_space_dim()

    def expm(self, dt, X_spatial=None):
        return self.base.expm(dt, X_spatial = X_spatial)

    def Q(self, dt_k, A_k, P_inf, X_spatial=None):
        return self.base.Q(dt_k, A_k, P_inf, X_spatial=X_spatial)
    def P_inf(self, x, X_s, t):
        return self.base.P_inf(x, X_s, t)



    @property
    def base(self):
        return self.parent_kernel



class DummyDerivativeKernel(DerivativeKernel):
    """ This is hack so that we can construct a model with no time derivates """
    def __init__(
            self, 
            parent_kernel = None
        ):

        super(DummyDerivativeKernel, self).__init__(parent_kernel)

        self.parent_output_dim = 1
        self.d_computed = 1
        self.output_dim  = 1 

    def K(self, X1, X2):
        return self.parent_kernel.K(X1, X2)

    def _K_from_fn(self, X1, X2, var_fn):
        return self.K(X1, X2)

class FirstOrderDerivativeKernel(DerivativeKernel):
    """ Construct FirstOrderDerivative kernel for the input index provided """
    def __init__(
            self, 
            parent_kernel = None,
            input_index: int = 0,
            parent_output_dim: int = 1
        ):

        super(FirstOrderDerivativeKernel, self).__init__(parent_kernel)

        self.parent_output_dim = parent_output_dim
        self.d_computed = 2
        self.output_dim  = self.d_computed * self.parent_output_dim 
        self.input_index = input_index

    def _compute_derivatives(self, x1, x2, var_fn):
        K = FirstOrderDerivativeKernel_compute_derivatives(x1, x2, var_fn, self.input_index, self.d_computed)


        chex.assert_shape(K, [self.output_dim, self.output_dim])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        # K is in data-diff format -- convert to diff-data format
        # forces B - D - data format
        K_reshaped = np.block([
            [
                K[:, :, d1, d2]
                for d2 in range(self.output_dim)
            ]
            for d1 in range(self.output_dim)
        ])

        return K_reshaped


class SecondOrderDerivativeKernel(DerivativeKernel):
    """ Construct SecondOrderDerivativeKernel kernel for the input index provided """
    def __init__(
            self, 
            parent_kernel = None,
            input_index: int = 0,
            parent_output_dim: int = 1
        ):

        super(SecondOrderDerivativeKernel, self).__init__(parent_kernel)

        self.parent_output_dim = parent_output_dim
        self.d_computed = 3
        self.output_dim  = self.d_computed * self.parent_output_dim 
        self.input_index = input_index

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let T to denote a differential operator: d/dt

        Let D = 
            I., .(T), .(T^2), 
            (T).,    (T).(T),    (T).(T^2)        
            (T)^2.,  (T)^2.(T),  (T)^2.(T^2) 

        be a matrix of linear operators (where . denotes the operator input, and we ignore transposes). 

        Then The full joint kernel is given by (ignoring transposes, and abusing the kronecker product notation):

            K ⊗ D

        When K is scalar this is given as

            K,       K(T),       K(T^2)             
            (T)K,    (T)K(T),    (T)K(T^2)        
            (T)^2K,  (T)^2K(T),  (T)^2K(T^2)   

        This means that the output is ordered by [f_1, (T)f_1, (T^2)f_1, ..., f_B, (T)f_B, (T^2)f_B]^T.
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
        # (T^2)K
        #  B x B x D x D
        res20 = hessian(k, argnums=(0))(x1, x2)

        # Computes
        # K(T^2)
        #  B x B x D x D
        res02 = hessian(k, argnums=(1))(x1, x2)

        # Computes
        # (T)K(T)
        #  B x B x D x D
        res11 = jacfwd(jacfwd(k, argnums=(0)), argnums=(1))(x1, x2)

        # Computes
        # (T)K(T^2)
        #  B x B x D x D x D
        res12 = hessian(jacfwd(k, argnums=(0)), argnums=(1))(x1, x2)
        # (T^2)K(T)
        #  B x B x D x D x D
        res21 = hessian(jacfwd(k, argnums=(1)), argnums=(0))(x1, x2)

        # arg 0 are the first dim, arg1 are the final
        # (T^2)K(T^2)
        #  B x B x D x D x D x D
        res22 = hessian(hessian(k, argnums=(0)), argnums=(1))(x1, x2)

        # Construct full matrix
        # K,       K(T),       K(T^2)
        # (T)K,    (T)K(T),    (T)K(T^2)
        # (T)^2K,  (T)^2K(T),  (T)^2K(T^2)


        # for a given B_i, B_j compute the derivate kernels
        def get_K(i, j):
            return np.array([
                [res00[i, j],       res01[i, j, self.input_index],        res02[i, j, self.input_index, self.input_index]], # f
                [res10[i, j, self.input_index],    res11[i, j, self.input_index, self.input_index],     res12[i, j, self.input_index, self.input_index, self.input_index]], # df/dt
                [res20[i, j, self.input_index, self.input_index], res21[i, j, self.input_index, self.input_index, self.input_index],  res22[i, j, self.input_index, self.input_index, self.input_index, self.input_index]], # d^2f/dt^2
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
        chex.assert_shape(K, [B*self.d_computed, B*self.d_computed])
        chex.assert_shape(K, [self.output_dim, self.output_dim])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        # K is in data-diff format -- convert to diff-data format
        K_reshaped = np.block([
            [
                K[:, :, d1, d2]
                for d2 in range(self.output_dim)
            ]
            for d1 in range(self.output_dim)
        ])

        return K_reshaped

class SecondOrderOnlyDerivativeKernel(DerivativeKernel):
    """ Only Construct SecondOrderDerivativeKernel kernel for the input index provided """
    def __init__(
            self, 
            parent_kernel = None,
            input_index: int = 0,
            parent_output_dim: int = 1
        ):

        super(SecondOrderOnlyDerivativeKernel, self).__init__(parent_kernel)

        self.parent_output_dim = parent_output_dim
        self.d_computed = 2
        self.output_dim  = self.d_computed * self.parent_output_dim 
        self.input_index = input_index

    def _compute_derivatives(self, x1, x2, var_fn):
        K = SecondOrderOnlyDerivativeKernel_compute_derivatives(x1, x2, var_fn, self.input_index, self.d_computed)
        chex.assert_shape(K, [self.output_dim, self.output_dim])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        # K is in data-diff format -- convert to diff-data format
        K_reshaped = np.block([
            [
                K[:, :, d1, d2]
                for d2 in range(self.output_dim)
            ]
            for d1 in range(self.output_dim)
        ])

        return K_reshaped

class RemoveDiffDim(DerivativeKernel):
    def __init__(
            self, 
            parent_kernel = None,
            input_index: int = 0
        ):

        super(RemoveDiffDim, self).__init__(parent_kernel)

        self.parent_kernel = parent_kernel
        self.parent_output_dim = self.parent_kernel.output_dim
        self.d_computed = self.parent_output_dim - 1
        self.output_dim  = self.d_computed  
        self.input_index = input_index
        self.index_to_keep = list(set(range(self.parent_output_dim))-set([self.input_index]))

    def K(self, X1, X2):
        K_parent = self.parent_kernel.K(X1, X2)
        N1 = X1.shape[0]
        N2 = X2.shape[0]

        # index to remove
        ind_1 = N1 * self.input_index
        ind_2 = N2 * self.input_index
        #K_parent = K_parent[~(ind_1-1):(ind_1+N1)]

        N1_index = np.arange(N1)[:, None]
        N2_index = np.arange(N2)[:, None]

        N1_index = np.tile(np.array(self.index_to_keep)[None, :], [N1, 1])*N1 + np.tile(np.arange(N1)[:, None], [1, self.output_dim])
        N1_index = np.transpose(N1_index).reshape(N1*self.output_dim)

        N2_index = np.tile(np.array(self.index_to_keep)[None, :], [N2, 1])*N2 + np.tile(np.arange(N2)[:, None], [1, self.output_dim])
        N2_index = np.transpose(N2_index).reshape(N2*self.output_dim)

        K_res = K_parent[N1_index][:, N2_index]

        chex.assert_rank(K_res, 2)
        chex.assert_shape(K_res, [N1*self.output_dim, N2*self.output_dim])

        return K_res

# ================== special cases ==============
# required as stacking DerivativeKernel above can make jitting slow

class FirstOrderDerivativeKernel_1D(DerivativeKernel):
    def __init__(
            self, 
            parent_kernel = None,
        ):

        super(FirstOrderDerivativeKernel_1D, self).__init__(parent_kernel)
        self.output_dim = 2

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let x1 have columns denotes by [t] then we use 
            Tto denote the differential operators d/dt

        The full joint kernel is given by (ignoring transposes):

            K,       K(T)          
            (T)K,    (T)K(T)       

        """
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]

        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # K(T)
        res01 = grad(k, argnums=(1))(x1, x2)


        # Computes
        # (T)K(T)
        res11 = jacfwd(grad(k, argnums=(0)), argnums=(1))(x1, x2)

        # Construct full matrix
        # K,       K(T))
        # (T)K,    (T)K(T)

        K = np.array([
            [res00,       res01[0]], # f
            [res10[0],    res11[0, 0]], # df/dt
        ])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1]],
            [K[:, :, 1, 0], K[:, :, 1, 1]],
        ])

        return K_reshaped

class FirstOrderKroneckerDerivativeKernel_2D(DerivativeKernel):
    """
    Compute first order derivates in x1 \kron x2 format.
    """
    def __init__(
            self, 
            parent_kernel = None,
        ):

        super(FirstOrderDerivativeKernel_2D, self).__init__(parent_kernel)
        # f, df/dx1, df/dx2, d^2f/(dx1 dx2)
        self.output_dim = 4

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let x1 have columns denotes by [t] then we use 
            Tto denote the differential operators d/dt

        The full joint kernel is given by (ignoring transposes):

            K,       K(T),      K(S),        K(S)(T)
            (T)K,    (T)K(T),   (T)K(S),     (T)K(S)(T)
            (S)K,    (S)K(T),   (S)K(S),     (S)K(S)(T)
            (T)(S)K, (T)(S)K(T), (T)(S)K(S), (T)(S)K(S)(T)

        """
        raise NotImplementedError()
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]

        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K, (S)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # [K(T), K(S)]
        res01 = grad(k, argnums=(1))(x1, x2)

        # Computes
        # [0][0]
        #  (T)(T)K, (T)(S)K
        #  (S)(T)K,    (S)(S)K

        # [0][1]
        #  (T)K(T), (T)K(S)
        #  (S)K(T),    (S)K(S)

        # [1][0]
        #  (T)K(T), (T)K(S)
        #  (S)K(T),    (S)K(S)

        # [1][1]
        #  K(T)(T),  K(S)(T)
        #  K(T) (S), K(S)(S)

        res11 = hessian(k, argnums=(0, 1))(x1, x2)


        res22 = hessian(hessian(k, argnums=(0, 1)), argnums=(0, 1))(x1, x2)

        # Computes (t)(s) k . or . k (s)(t) 
        # Args:
        #  inner_axis: which axis to take the first derivate wrt
        #  outer_axis: which axis to take the first derivate wrt
        #  inner: which dimension we want to take from the first derivate
        #  outer: which dimension we want to take from the second derivate
        joint_diff = lambda fn: lambda inner_axis, outer_axis, inner, outer: jacfwd(
            lambda a1, a2: grad(
                fn, argnums=(inner_axis)
            )(a1, a2)[inner], 
            argnums=(outer_axis)
        )(x1, x2)[outer]

        res_st_k = joint_diff(0, 0, 0, 1)
        res_k_st = joint_diff(1, 1, 1, 0)


        # Computes
        #  (T)(T)K, (T)(S)K
        #  (T)(S)K,    (S)(S)K

        #  (T)K(T),   (T)K(S)
        #  (S)K(T),   (S)K(S)

        res21 = jacfwd(grad(k, argnums=(0)), argnums=(0, 1))(x1, x2)

        # Computes
        #  K(T)(T), K(S)(T)
        #  K(S)(T), K(S)(S)
        res12 = jacfwd(grad(k, argnums=(1)), argnums=(0, 1))(x1, x2)


        # Construct full matrix
        # K,       K(T))
        # (T)K,    (T)K(T)

        K = np.array([
            [res00,       res01[0]], # f
            [res10[0],    res11[0, 0]], # df/dt
        ])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1]],
            [K[:, :, 1, 0], K[:, :, 1, 1]],
        ])

        return K_reshaped


class SecondOrderDerivativeKernel_1D(DerivativeKernel):
    def __init__(
            self, 
            parent_kernel = None,
        ):

        super(SecondOrderDerivativeKernel_1D, self).__init__(parent_kernel)
        self.output_dim = 3

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let x1 have columns denotes by [t] then we use 
            Tto denote the differential operators d/dt

        The full joint kernel is given by (ignoring transposes):

            K,       K(T),       K(T^2)             
            (T)K,    (T)K(T),    (T)K(T^2)        
            (T)^2K,  (T)^2K(T),  (T)^2K(T^2)   

        """
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]

        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # K(T)
        res01 = grad(k, argnums=(1))(x1, x2)

        # (T^2)K
        res20 = hessian(k, argnums=(0))(x1, x2)

        # K(T^2)
        res02 = hessian(k, argnums=(1))(x1, x2)

        # Computes
        # (T)K(T)
        res11 = jacfwd(grad(k, argnums=(0)), argnums=(1))(x1, x2)

        # Computes
        # (T)K(T^2)
        res12 = hessian(grad(k, argnums=(0)), argnums=(1))(x1, x2)
        # (T^2)K(T)
        res21 = hessian(grad(k, argnums=(1)), argnums=(0))(x1, x2)

        # arg 0 are the first dim, arg1 are the final
        # (T^2)K(T^2)
        res22 = hessian(hessian(k, argnums=(0)), argnums=(1))(x1, x2)

        # Construct full matrix
        # K,       K(T),       K(T^2)
        # (T)K,    (T)K(T),    (T)K(T^2)
        # (T)^2K,  (T)^2K(T),  (T)^2K(T^2)

        K = np.array([
            [res00,       res01[0],        res02[0, 0]], # f
            [res10[0],    res11[0, 0],     res12[0, 0, 0]], # df/dt
            [res20[0][0], res21[0, 0, 0],  res22[0, 0, 0, 0]], # d^2f/dt^2
        ])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1], K[:, :, 0, 2]],
            [K[:, :, 1, 0], K[:, :, 1, 1], K[:, :, 1, 2]],
            [K[:, :, 2, 0], K[:, :, 2, 1], K[:, :, 2, 2]],
        ])

        return K_reshaped


class SecondOrderDerivativeKernel_2D(DerivativeKernel):
    def __init__(
            self, 
            parent_kernel = None
        ):

        super(SecondOrderDerivativeKernel_2D, self).__init__(parent_kernel)
        self.output_dim = 5

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let x1 have columns denotes by [t, s1] then we use 
            T, S1 to denote the differential operators d/dt, d/ds1

        The full joint kernel is given by (ignoring transposes):

            K,       K(T),       K(T^2),       K(S1),       K(S1^2),      
            (T)K,    (T)K(T),    (T)K(T^2),    (T)K(S1),    (T)K(S1^2),    
            (T)^2K,  (T)^2K(T),  (T)^2K(T^2),  (T)^2K(S1),  (T)^2K(S1^2),  
            (S1)K,   (S1)K(T),   (S1)K(T^2),   (S1)K(S1),   (S1)K(S1^2),   
            (S1^2)K, (S1^2)K(T), (S1^2)K(T^2), (S1^2)K(S1), (S1^2)K(S1^2), 

        """
        # fix shapes
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]

        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K, (S1)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # [K(T), K(S1)]^T
        res01 = grad(k, argnums=(1))(x1, x2)

        # (T^2)K, (T)(S1)K
        # (T)(S1)K, (S1^2)K
        res20 = hessian(k, argnums=(0))(x1, x2)

        # K(T^2), K(T)(S1)
        # K(T)(S1), K(S1^2)
        res02 = hessian(k, argnums=(1))(x1, x2)

        # Computes
        # (T)K(T), (T)K(S1)
        # (S1)K(T), (S1)K(S1)
        res11 = jacfwd(grad(k, argnums=(0)), argnums=(1))(x1, x2)

        # Computes
        # (T)K(T^2),   (T)K(T)(S1)
        # (T)K(T)(S1), (T)K(S1^2)
        #-
        # (S1)K(T^2),   (S1)K(T)(S1)
        # (S1)K(T)(S1), (S1)K(S1^2)
        #-
        # (S2)K(T^2),   (S2)K(T)(S1)
        # (S2)K(T)(S1), (S2)K(S1^2)
        res12 = hessian(grad(k, argnums=(0)), argnums=(1))(x1, x2)
        res21 = hessian(grad(k, argnums=(1)), argnums=(0))(x1, x2)

        # arg 0 are the first dim, arg1 are the final
        res22 = hessian(hessian(k, argnums=(0)), argnums=(1))(x1, x2)

        # Construct full matrix
        # K,       K(T),       K(T^2),       K(S1),       K(S1^2)
        # (T)K,    (T)K(T),    (T)K(T^2),    (T)K(S1),    (T)K(S1^2)
        # (T)^2K,  (T)^2K(T),  (T)^2K(T^2),  (T)^2K(S1),  (T)^2K(S1^2)
        # (S1)K,   (S1)K(T),   (S1)K(T^2),   (S1)K(S1),   (S1)K(S1^2)
        # (S1^2)K, (S1^2)K(T), (S1^2)K(T^2), (S1^2)K(S1), (S1^2)K(S1^2)

        K = np.array([
            [res00,       res01[0],        res02[0, 0],       res01[1],       res02[1, 1]], # f
            [res10[0],    res11[0, 0],     res12[0, 0, 0],    res11[0, 1],    res12[0, 1, 1]], # df/dt
            [res20[0][0], res21[0, 0, 0],  res22[0, 0, 0, 0], res21[1, 0, 0], res22[0, 0, 1, 1]], # d^2f/dt^2
            [res10[1],    res11[1, 0],     res12[1, 0, 0],    res11[1, 1],    res12[1, 1, 1]], # df / dx1
            [res20[1][1], res21[0, 1, 1],  res22[0, 0, 1, 1], res21[1, 1,1],  res22[1, 1, 1, 1]]# d^2f / dx1^2
        ])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1], K[:, :, 0, 2], K[:, :, 0, 3], K[:, :, 0, 4]],
            [K[:, :, 1, 0], K[:, :, 1, 1], K[:, :, 1, 2], K[:, :, 1, 3], K[:, :, 1, 4]],
            [K[:, :, 2, 0], K[:, :, 2, 1], K[:, :, 2, 2], K[:, :, 2, 3], K[:, :, 2, 4]],
            [K[:, :, 3, 0], K[:, :, 3, 1], K[:, :, 3, 2], K[:, :, 3, 3], K[:, :, 3, 4]],
            [K[:, :, 4, 0], K[:, :, 4, 1], K[:, :, 4, 2], K[:, :, 4, 3], K[:, :, 4, 4]]
        ])

        return K_reshaped

class SecondOrderSpaceFirstOrderTimeDerivativeKernel_2D(SecondOrderDerivativeKernel_2D):
    def __init__(
            self, 
            parent_kernel = None
        ):

        super(SecondOrderSpaceFirstOrderTimeDerivativeKernel_2D, self).__init__(parent_kernel)
        self.output_dim = 3

    def _K_from_fn(self, X1, X2, var_fn):
        Kxx = var_fn(X1, X2)

        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0],  K[:, :, 0, 1], K[:, :, 0, 4]],
            [K[:, :, 1, 0],  K[:, :, 1, 1], K[:, :, 1, 4]],
            [K[:, :, 4, 0],  K[:, :, 4, 1], K[:, :, 4, 4]]
        ])

        return K_reshaped


class SecondOrderDerivativeKernel_3D(DerivativeKernel):
    def __init__(
            self, 
            parent_kernel = None
        ):

        super(SecondOrderDerivativeKernel_3D, self).__init__(parent_kernel)
        self.output_dim = 7

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let x1 have columns denotes by [t, s1, s2] then we use 
            T, S1, S2 to denote the differential operators d/dt, d/ds1, d/ds2 

        The full joint kernel is given by (ignoring transposes):

            K,       K(T),       K(T^2),       K(S1),       K(S1^2),       K(S2),       K(S2^2)
            (T)K,    (T)K(T),    (T)K(T^2),    (T)K(S1),    (T)K(S1^2),    (T)K(S2),    (T)K(S2^2)
            (T)^2K,  (T)^2K(T),  (T)^2K(T^2),  (T)^2K(S1),  (T)^2K(S1^2),  (T)^2K(S2),  (T)^2K(S2^2)
            (S1)K,   (S1)K(T),   (S1)K(T^2),   (S1)K(S1),   (S1)K(S1^2),   (S1)K(S2),   (S1)K(S2^2)
            (S1^2)K, (S1^2)K(T), (S1^2)K(T^2), (S1^2)K(S1), (S1^2)K(S1^2), (S1^2)K(S2), (S1^2)K(S2^2)
            (S2)K,   (S2)K(T),   (S2)K(T^2),   (S2)K(S1),   (S2)K(S1^2),   (S2)K(S2),   (S2)K(S2^2)
            (S2)^2K, (S2)^2K(T), (S2)^2K(T^2), (S2)^2K(S1), (S2)^2K(S1^2), (S2)^2K(S2), (S2)^2K(S2^2)

        """
        # fix shapes
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]


        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K, (S1)K, (S2)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # [K(T), K(S1), K(S2)]^T
        res01 = grad(k, argnums=(1))(x1, x2)

        # (T^2)K, (T)(S1)K, (T)(S2)K
        # (T)(S1)K, (S1^2)K, (S1)(S2)K
        # (T)(S2)K, (S1)(S2)K, (S2^2)K
        res20 = hessian(k, argnums=(0))(x1, x2)

        # K(T^2), K(T)(S1), K(T)(S2)
        # K(T)(S1), K(S1^2), K(S1)(S2)
        # K(T)(S2), K(S1)(S2), K(S2^2)
        res02 = hessian(k, argnums=(1))(x1, x2)

        # Computes
        # (T)K(T), (T)K(S1), (T)K(S2)
        # (S1)K(T), (S1)K(S1), (S1)K(S2)
        # (S2)K(T)(S2), (S2)K(S1), (S2)K(S2)
        res11 = jacfwd(grad(k, argnums=(0)), argnums=(1))(x1, x2)

        # Computes
        # (T)K(T^2),   (T)K(T)(S1),  (T)K(T)(S2)
        # (T)K(T)(S1), (T)K(S1^2),   (T)K(S1)(S2)
        # (T)K(T)(S2), (T)K(S1)(S2), (T)K(S2^2)
        #-
        # (S1)K(T^2),   (S1)K(T)(S1),  (S1)K(T)(S2)
        # (S1)K(T)(S1), (S1)K(S1^2),   (S1)K(S1)(S2)
        # (S1)K(T)(S2), (S1)K(S1)(S2), (S1)K(S2^2)
        #-
        # (S2)K(T^2),   (S2)K(T)(S1),  (S2)K(T)(S2)
        # (S2)K(T)(S1), (S2)K(S1^2),   (S2)K(S1)(S2)
        # (S2)K(T)(S2), (S2)K(S1)(S2), (S2)K(S2^2)
        res12 = hessian(grad(k, argnums=(0)), argnums=(1))(x1, x2)
        res21 = hessian(grad(k, argnums=(1)), argnums=(0))(x1, x2)

        # arg 0 are the first dim, arg1 are the final
        res22 = hessian(hessian(k, argnums=(0)), argnums=(1))(x1, x2)

        # Construct full matrix
        # K,       K(T),       K(T^2),       K(S1),       K(S1^2),       K(S2),       K(S2^2)
        # (T)K,    (T)K(T),    (T)K(T^2),    (T)K(S1),    (T)K(S1^2),    (T)K(S2),    (T)K(S2^2)
        # (T)^2K,  (T)^2K(T),  (T)^2K(T^2),  (T)^2K(S1),  (T)^2K(S1^2),  (T)^2K(S2),  (T)^2K(S2^2)
        # (S1)K,   (S1)K(T),   (S1)K(T^2),   (S1)K(S1),   (S1)K(S1^2),   (S1)K(S2),   (S1)K(S2^2)
        # (S1^2)K, (S1^2)K(T), (S1^2)K(T^2), (S1^2)K(S1), (S1^2)K(S1^2), (S1^2)K(S2), (S1^2)K(S2^2)
        # (S2)K,   (S2)K(T),   (S2)K(T^2),   (S2)K(S1),   (S2)K(S1^2),   (S2)K(S2),   (S2)K(S2^2)
        # (S2)^2K, (S2)^2K(T), (S2)^2K(T^2), (S2)^2K(S1), (S2)^2K(S1^2), (S2)^2K(S2), (S2)^2K(S2^2)

        K = np.array([
            [res00,       res01[0],        res02[0, 0],       res01[1],       res02[1, 1],       res01[2],       res02[2, 2]], # f
            [res10[0],    res11[0, 0],     res12[0, 0, 0],    res11[0, 1],    res12[0, 1, 1],    res11[0, 2],    res12[0, 2, 2]], # df/dt
            [res20[0][0], res21[0, 0, 0],  res22[0, 0, 0, 0], res21[1, 0, 0], res22[0, 0, 1, 1], res21[2, 0, 0], res22[0, 0, 2, 2]], # d^2f/dt^2
            [res10[1],    res11[1, 0],     res12[1, 0, 0],    res11[1, 1],    res12[1, 1, 1],    res11[1, 2],    res12[1, 2, 2]], # df / dx1
            [res20[1][1], res21[0, 1, 1],  res22[0, 0, 1, 1], res21[1, 1,1],  res22[1, 1, 1, 1], res21[2, 1, 1], res22[1, 1, 2, 2]],# d^2f / dx1^2
            [res10[2],    res11[2, 0],     res12[2, 0, 0],    res11[2,1],     res12[2, 1, 1],    res11[2, 2],    res12[2, 2, 2]],# df / dx2
            [res20[2][2], res21[0, 2, 2],  res22[0, 0, 2, 2], res21[1, 2, 2], res22[2, 2, 1, 1], res21[2, 2, 2], res22[2, 2, 2, 2]] # d^2f / dx2^2
        ])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1], K[:, :, 0, 2], K[:, :, 0, 3], K[:, :, 0, 4], K[:, :, 0, 5], K[:, :, 0, 6]],
            [K[:, :, 1, 0], K[:, :, 1, 1], K[:, :, 1, 2], K[:, :, 1, 3], K[:, :, 1, 4], K[:, :, 1, 5], K[:, :, 1, 6]],
            [K[:, :, 2, 0], K[:, :, 2, 1], K[:, :, 2, 2], K[:, :, 2, 3], K[:, :, 2, 4], K[:, :, 2, 5], K[:, :, 2, 6]],
            [K[:, :, 3, 0], K[:, :, 3, 1], K[:, :, 3, 2], K[:, :, 3, 3], K[:, :, 3, 4], K[:, :, 3, 5], K[:, :, 3, 6]],
            [K[:, :, 4, 0], K[:, :, 4, 1], K[:, :, 4, 2], K[:, :, 4, 3], K[:, :, 4, 4], K[:, :, 4, 5], K[:, :, 4, 6]],
            [K[:, :, 5, 0], K[:, :, 5, 1], K[:, :, 5, 2], K[:, :, 5, 3], K[:, :, 5, 4], K[:, :, 5, 5], K[:, :, 5, 6]],
            [K[:, :, 6, 0], K[:, :, 6, 1], K[:, :, 6, 2], K[:, :, 6, 3], K[:, :, 6, 4], K[:, :, 6, 5], K[:, :, 6, 6]],
        ])

        return K_reshaped


class FirstOrderDerivativeKernel_3D(DerivativeKernel):
    def __init__(
            self, 
            parent_kernel = None
        ):

        super(FirstOrderDerivativeKernel_3D, self).__init__(parent_kernel)
        self.output_dim = 4

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let x1 have columns denotes by [t, s1, s2] then we use 
            T, S1, S2 to denote the differential operators d/dt, d/ds1, d/ds2 

        The full joint kernel is given by (ignoring transposes):

            K,       K(T),       K(S1),       K(S2)
            (T)K,    (T)K(T),    (T)K(S1),    (T)K(S2)
            (S1)K,   (S1)K(T),   (S1)K(S1),   (S1)K(S2)
            (S2)K,   (S2)K(T),   (S2)K(S1),   (S2)K(S2)

        """
        # fix shapes
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]


        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K, (S1)K, (S2)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # [K(T), K(S1), K(S2)]^T
        res01 = grad(k, argnums=(1))(x1, x2)

        # Computes
        # (T)K(T), (T)K(S1), (T)K(S2)
        # (S1)K(T), (S1)K(S1), (S1)K(S2)
        # (S2)K(T)(S2), (S2)K(S1), (S2)K(S2)
        res11 = jacfwd(grad(k, argnums=(0)), argnums=(1))(x1, x2)

        # Construct full matrix
        # K,       K(T),       K(S1),       K(S2)
        # (T)K,    (T)K(T),    (T)K(S1),    (T)K(S2)
        # (S1)K,   (S1)K(T),   (S1)K(S1),   (S1)K(S2)
        # (S2)K,   (S2)K(T),   (S2)K(S1),   (S2)K(S2)

        K = np.array([
            [res00,       res01[0],        res01[1],       res01[2]], # f
            [res10[0],    res11[0, 0],     res11[0, 1],    res11[0, 2]], # df/dt
            [res10[1],    res11[1, 0],     res11[1, 1],    res11[1, 2]], # df / dx1
            [res10[2],    res11[2, 0],     res11[2,1],     res11[2, 2]],# df / dx2
        ])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1], K[:, :, 0, 2], K[:, :, 0, 3]],
            [K[:, :, 1, 0], K[:, :, 1, 1], K[:, :, 1, 2], K[:, :, 1, 3]],
            [K[:, :, 2, 0], K[:, :, 2, 1], K[:, :, 2, 2], K[:, :, 2, 3]],
            [K[:, :, 3, 0], K[:, :, 3, 1], K[:, :, 3, 2], K[:, :, 3, 3]]
        ])

        return K_reshaped


class FirstOrderDerivativeKernel_2D(DerivativeKernel):
    """
    Compute first order derivates 
    """
    def __init__(
            self, 
            parent_kernel = None,
            input_index = 0
        ):

        super(FirstOrderDerivativeKernel_2D, self).__init__(parent_kernel)
        # f, df/dx1, df/dx2
        self.output_dim = 3
        # TODO: this is a bit confusing
        self.d_computed = 3
        self.input_index = input_index

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let x1 have columns denotes by [t, s1] then we use 
            T, S1 to denote the differential operators d/dt, d/ds1

        The full joint kernel is given by (ignoring transposes):

            K,       K(T),      K(S1),       
            (T)K,    (T)K(T),   (T)K(S1),      
            (S1)K,   (S1)K(T),  (S1)K(S1),   

        """
        # fix shapes
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])[0, 0]

        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
        res00 = k(x1, x2)

        # D dimensional jacobian vector
        # [(T)K, (S1)K]
        res10 = grad(k, argnums=(0))(x1, x2)
        # [K(T), K(S1)]^T
        res01 = grad(k, argnums=(1))(x1, x2)

        # Computes
        # (T)K(T), (T)K(S1)
        # (S1)K(T), (S1)K(S1)
        res11 = jacfwd(grad(k, argnums=(0)), argnums=(1))(x1, x2)

        # Construct full matrix
        # K,       K(T),       K(S1)
        # (T)K,    (T)K(T),    (T)K(S1)
        # (S1)K,   (S1)K(T),   (S1)K(S1)

        K = np.array([
            [res00,       res01[0+self.input_index],        res01[1+self.input_index]], # f
            [res10[0+self.input_index],    res11[0+self.input_index, 0+self.input_index],     res11[0+self.input_index, 1+self.input_index]], # df/dt
            [res10[1+self.input_index],    res11[1+self.input_index, 0+self.input_index],     res11[1+self.input_index, 1+self.input_index]], # df / dx1
        ])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0], K[:, :, 0, 1], K[:, :, 0, 2]],
            [K[:, :, 1, 0], K[:, :, 1, 1], K[:, :, 1, 2]],
            [K[:, :, 2, 0], K[:, :, 2, 1], K[:, :, 2, 2]],
        ])

        return K_reshaped

class SecondOrderOnlyDerivativeKernel_2D(DerivativeKernel):
    """
    Compute second order derivates 
    """
    def __init__(
            self, 
            parent_kernel = None,
            input_index = 0,
            parent_output_dim = 1
        ):

        super(SecondOrderOnlyDerivativeKernel_2D, self).__init__(parent_kernel)
        # f, df2/dx12, d2f/dx22
        self.d_computed = 3
        self.input_index = input_index
        self.parent_output_dim = parent_output_dim
        self.output_dim = self.d_computed * self.parent_output_dim

    def _compute_derivatives(self, x1, x2, var_fn):
        """
        Let x1 have columns denotes by [t, s1] then we use 
            T, S1 to denote the differential operators d/dt, d/ds1

        The full joint kernel is given by (ignoring transposes):

            K,       K(T)^2,      K(S1)^2,       
            (T)^2K,    (T)^2K(T)^2,   (T)^2K(S1)^2,      
            (S1)^2K,   (S1)^2K(T)^2,  (S1)^2K(S1)^2,   

        """
        # fix shapes
        k = lambda x1, x2: var_fn(x1[None, ...], x2[None, ...])

        # compute blocks

        # variable name notation
        # res<x1 diff_order><x2 diff order>
        #scalar
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
        # K,       K(T)^2,       K(S1)^2
        # (T)^2 K,    (T)^2 K (T)^2,    (T)^2K(S1)^2
        # (S1) ^2K,   (S1)^2 K (T)^2,   (S1)^2K(S1)^2

        id0 = self.input_index
        id1 = self.input_index+1


        # for a given B_i, B_j compute the derivate kernels
        def get_K(i, j):
            return np.array([
                [res00[i, j],       res02[i, j][id0][id0],        res02[i, j][id1][id1]], # f
                [res20[i, j][id0][id0],    res22[i, j][id0, id0][id0, id0],     res22[i, j][id0, id0][id1, id1]], # df/dt
                [res20[i, j][id1][id1],    res22[i, j][id1, id1][id0, id0],     res22[i, j][id1, id1][id1, id1]], # df / dx1
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
        chex.assert_shape(K, [B*self.d_computed, B*self.d_computed])
        chex.assert_shape(K, [self.output_dim, self.output_dim])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        K = jax.vmap(k2, (0, None))(X1, X2)

        # K is in data-diff format -- convert to diff-data format
        K_reshaped = np.block([
            [
                K[:, :, d1, d2]
                for d2 in range(self.output_dim)
            ]
            for d1 in range(self.output_dim)
        ])

        return K_reshaped

class SecondOrderSpaceFirstOrderTimeDerivativeKernel_3D(SecondOrderDerivativeKernel_3D):
    """
    Let x1 have columns denotes by [t, s1, s2] then we use 
        T, S1 to denote the differential operators d/dt, d/ds1

    The full joint kernel is given by (ignoring transposes):

        K,         K(T),        K(S1)^2,         K(S2)^2
        (T)K,      (T)K(T),     (T)K(S1)^2,      (T)K(S2)^2 
        (S1)^2K,   (S1)^2K(T),  (S1)^2K(S1)^2,   (S1)^2K(S2)^2
        (S2)^2K,   (S2)^2K(T),  (S2)^2K(S1)^2,   (S2)^2K(S2)^2

    """
    def __init__(
        self, 
        parent_kernel = None
    ):

        super(SecondOrderSpaceFirstOrderTimeDerivativeKernel_3D, self).__init__(parent_kernel)
        self.output_dim = 4

    def _K_from_fn(self, X1, X2, var_fn):
        Kxx = var_fn(X1, X2)

        def k2(x1, X2):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, var_fn)

        # [K,       K(T),       K(T^2),       K(S1),       K(S1^2),       K(S2),       K(S2^2)]
        K = jax.vmap(k2, (0, None))(X1, X2)

        #return K[:, :, 0, 0]
        #reshape to NxN
        K_reshaped =  np.block([
            [K[:, :, 0, 0],  K[:, :, 0, 1], K[:, :, 0, 4], K[:, :, 0, 6]],
            [K[:, :, 1, 0],  K[:, :, 1, 1], K[:, :, 1, 4], K[:, :, 1, 6]],
            [K[:, :, 4, 0],  K[:, :, 4, 1], K[:, :, 4, 4], K[:, :, 4, 6]],
            [K[:, :, 6, 0],  K[:, :, 6, 1], K[:, :, 6, 4], K[:, :, 6, 6]]
        ])

        return K_reshaped


# =================== CLOSED FORMS ==========================

class ClosedFormRBFFirstOrderDerivativeKernel(FirstOrderDerivativeKernel):
    """
    parent kernel must be RBF

    NOTE: this should only be used for hierachical models!!
    """

    def __init__(
            self, 
            parent_kernel = None,
            input_index: int = 0,
            parent_output_dim: int = 1
        ):

        super(ClosedFormRBFFirstOrderDerivativeKernel, self).__init__(parent_kernel, input_index, parent_output_dim)


    def _compute_derivatives(self, x1, x2, parent_kernel):
        #B x B
        k = lambda x1, x2: parent_kernel.K(x1[None, ...], x2[None, ...])

        # variable name notation
        # res<x1 diff_order><x2 diff order>

        ls = parent_kernel.lengthscales

        # Computes
        # [K]
        res00 = np.squeeze(k(x1, x2))

        _x1, _x2 = x1[self.input_index], x2[self.input_index]

        # -1 because RBF is only defined on the spatial part, or if not this will be negative one so it will select the correct ls
        _ls = np.squeeze(ls)

        # Computes
        # [(T)K]
        res10 = (1/(_ls**2)) * (_x1-_x2) * res00

        # Computes
        # K(T)
        # B x B x D
        res01 = (1/(_ls**2)) * (_x2-_x1) * res00

        # Computes
        # (T)K(T)
        tau = (_x1-_x2)*(_x2-_x1)
        gamma = (1/(_ls**2))
        res11 = gamma * (1 + gamma*tau) * res00
        #res11 = (1/(_ls**2)) * res00

        # Construct full matrix
        # K,       K(T)
        # (T)K,    (T)K(T)

        K =  np.array([
            [res00, res10],
            [res01, res11]
        ])

        return K

    def _K_from_fn(self, X1, X2, var_fn):
        # by construction var_fn is just the kernel function self.parent_kernel

        def k2(x1, X2, K_fn):
            return jax.vmap(self._compute_derivatives, (None, 0, None))(x1, X2, K_fn)

        K = jax.vmap(k2, (0, None, None))(X1, X2, self.parent_kernel)


        # K is in data-diff format -- convert to diff-data format
        # forces B - D - data format
        K_reshaped = np.block([
            [
                K[:, :, d1, d2]
                for d2 in range(self.output_dim)
            ]
            for d1 in range(self.output_dim)
        ])

        return K_reshaped

