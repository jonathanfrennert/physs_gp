import objax
import chex
import jax
import jax.numpy as np
from jax import jit

from jax.scipy.linalg import block_diag 
from abc import ABC
from abc import abstractmethod
from jax.numpy import vectorize
import typing
from typing import List, Optional, Union
from ..computation.parameter_transforms import inv_positive_transform, positive_transform
from ..utils.utils import ensure_array, ensure_float
from .. import Parameter
from functools import partial

from .ss_utils import space_time_state_space_rep


class Kernel(objax.Module):
    def __init__(
        self,
        input_dim: Optional[int] = 1,
        active_dims: Optional[np.ndarray] = None,
    ):
        chex.assert_type(input_dim, int)

        if active_dims is not None:
            active_dims = ensure_array(active_dims)
            chex.assert_rank(active_dims, 1)  # ensure array
            chex.assert_equal(input_dim, active_dims.shape[0])

        self.input_dim = input_dim
        self.active_dims = active_dims

    # kernel combinations
    def __add__(self, kern_2):
        return SumKernel(self, kern_2)

    def __mul__(self, kern_2):
        return ProductKernel(self, kern_2)

    def _apply_active_dim(self, X):
        if self.active_dims is None:
            X = X
        else:
            X = X[:, self.active_dims]

        return X

    @abstractmethod
    def K(self, X1: np.array, X2: np.array):
        chex.assert_rank([X1, X2], [2, 2])
        chex.assert_equal(X1.shape[1], X2.shape[1])

        _X1 = self._apply_active_dim(X1)
        _X2 = self._apply_active_dim(X2)

        return self._K(_X1, _X2)

    def _K(self, X1, X2):
        D = X1.shape[1]
        def _K_d2(x1, x2):
            chex.assert_shape(x1, [D])
            chex.assert_shape(x1, [D])

            k_d1_d2 = jax.vmap(self._K_scaler, in_axes=[0, 0])(x1, x2)
            chex.assert_shape(k_d1_d2, [D])

            k_xx =  np.prod(k_d1_d2)
            chex.assert_rank(k_xx, 0)

            return k_xx

        def _K_d1(x1, X2):
            #vectorised over first input
            return jax.vmap(_K_d2, in_axes=[None, 0], out_axes=0)(x1, X2)

        K = jax.vmap(_K_d1, in_axes=[0, None], out_axes=0)(X1, X2)

        chex.assert_equal(K.shape[0], X1.shape[0])
        chex.assert_equal(K.shape[1], X2.shape[0])

        return K

    @abstractmethod
    def K_diag(self, X: np.array):
        raise NotImplementedError()

    @property
    def base(self):
        return self


class CombinationKernel(Kernel):
    def __init__(self, k1: "Kernel", k2: "Kernel"):
        self.k1 = k1
        self.k2 = k2

    def __list__(self):
        if isinstance(self.k1, CombinationKernel):
            k1_arr = list(self.k1)
        else:
            k1_arr = [self.k1]

        if isinstance(self.k2, CombinationKernel):
            k2_arr = list(self.k2)
        else:
            k2_arr = [self.k2]

        return k1_arr + k2_arr

    def fix(self):
        self.k1.fix()
        self.k2.fix()

    def release(self):
        self.k1.release()
        self.k2.release()

    def __getitem__(self, index):
        """
        When a kernel is defined like:
            k = k1*k2*k3
        The equivalent kernel is:
            ProductKernel(k1, ProductKernel(k2, k3))
        """

        all_kernels = self.__list__()
        return all_kernels[index]


class SumKernel(CombinationKernel):
    def to_ss(self, X_spatial=None):
        k1_F, k1_L, k1_Qc, k1_H, m1_inf, k1_Pinf = self.k1.to_ss(X_spatial)
        k2_F, k2_L, k2_Qc, k2_H, m2_inf, k2_Pinf = self.k2.to_ss(X_spatial)

        F = block_diag(k1_F, k2_F)
        L = block_diag(k1_L, k2_L)
        Pinf = block_diag(k1_Pinf, k2_Pinf)
        Qc = block_diag(k1_Qc, k2_Qc)
        m_inf = np.vstack([m1_inf, m2_inf])
        H = np.hstack([k1_H, k2_H])

        return F, L, Qc, H, m_inf, Pinf

    def state_space_dim(self):
        return self.k1.state_space_dim()+self.k2.state_space_dim()

    def K(self, X1: np.array, X2: np.array):
        return self.k1.K(X1, X2) + self.k2.K(X1, X2)

    def K_diag(self, X1: np.array):
        return self.k1.K_diag(X1) + self.k2.K_diag(X1)

    def expm(self, dt, X_spatial=None):
        A1 = self.k1.expm(dt, X_spatial)
        A2 = self.k2.expm(dt, X_spatial)
        return block_diag(A1, A2)


class ProductKernel(CombinationKernel):
    def to_ss(self, X_spatial=None):
        k1_F, k1_L, k1_Qc, k1_H, m1_inf, k1_Pinf = self.k1.to_ss(X_spatial)
        k2_F, k2_L, k2_Qc, k2_H, m2_inf, k2_Pinf = self.k2.to_ss(X_spatial)

        I_1 = np.eye(k1_F.shape[0])
        I_2 = np.eye(k2_F.shape[0])

        F = np.kron(k1_F, I_1) + np.kron(I_2, k2_F)
        L = np.kron(k1_L, k2_L)
        Q = np.kron(k1_Qc, k2_Qc)
        Pinf = np.kron(k1_Pinf, k2_Pinf)
        # TODO -- need to check this
        m_inf = np.kron(m1_inf, m2_inf)
        H = np.kron(k1_H, k2_H)

        return F, L, Qc, H, Pinf

    def K(self, X1: np.array, X2: np.array):
        return self.k1.K(X1, X2) * self.k2.K(X1, X2)

    def K_diag(self, X1: np.array):
        return self.k1.K_diag(X1) * self.k2.K_diag(X1)

    def expm(self, dt, X_spatial=None):
        A1 = self.k1.expm(dt, X_spatial)
        A2 = self.k2.expm(dt, X_spatial)
        return np.kron(A1, A2)

class ConcatationKernel(CombinationKernel):
    def K(self, X1: np.array, X2: np.array):
        return np.array([self.k1.K(X1, X2), self.k2.K(X1, X2)])

    def K_diag(self, X1: np.array):

        return np.array([self.k1.K_diag(X1), self.k2.K_diag(X1)])

class MarkovKernel(Kernel):
    def state_space_dim(self):
        return self._state_space_dim

    def cf_to_ss_spatial(self, sparsity):
        raise NotImplementedError()

    def Q(self, dt, A_k, P_inf, X_spatial=None):
        # TODO: ONLY applies for stationary models
        return P_inf - A_k @  P_inf @ A_k.T



class SpatioTemporalSeperableKernel(MarkovKernel, ProductKernel):
    def state_space_dim(self):
        # only need return the time dim
        return self.k1.state_space_dim()

    def __init__(self, K_temporal, K_spatial, spatial_output_dim: int = 1, whiten_space = False, stationary=True):
        self.k1 = K_temporal
        self.k2 = K_spatial
        # when k2 is a DiffOp kernel this will change the output dim of space
        self.spatial_output_dim = spatial_output_dim
        self.whiten_space = whiten_space
        self.stationary = stationary

    def to_ss(self, X_spatial):
        # if the spatial kernel is a derivate kernel, just evaluate the base kernel
        #K_spatial = self.k2.base.K(X_spatial, X_spatial)
        # add on dummy time dimension
        X_spatial = np.hstack([np.zeros([X_spatial.shape[0], 1]), X_spatial])
        K_spatial = self.k2.K(X_spatial, X_spatial)

        if self.whiten_space:
            # whitened rep
            K_spatial = np.eye(K_spatial.shape[0])

        F, L, Qc, H, m_inf, Pinf = self.k1.to_ss()

        return space_time_state_space_rep(
            K_spatial,  F, L, Qc, H, m_inf, Pinf
        )

    def state_size(self):
        # only return the temporal state_size 
        return self.k1.state_size()

    def expm(self, dt, X_spatial):
        A_t = self.k1.expm(dt)

        eye = np.eye(self.spatial_output_dim*X_spatial.shape[0])

        A = np.kron(eye, A_t)

        return A


    def Q(self, dt, A_k, P_inf, X_spatial=None):
        if self.stationary:
            return P_inf - A_k @ P_inf @ A_k.T
        else:
            Q1 = self.k1.Q(dt, A_k, P_inf, X_spatial)

            X_spatial = np.hstack([np.zeros([X_spatial.shape[0], 1]), X_spatial])
            K_spatial = self.k2.K(X_spatial, X_spatial)
            return np.kron(K_spatial, Q1)

class WhiteNoiseKernel(Kernel):
    def __init__(
        self,
        input_dim: Optional[int] = 1, 
        active_dims: Optional[np.ndarray] = None
    ):
        super(WhiteNoiseKernel, self).__init__(input_dim, active_dims)

    def _K_scaler(self, x1, x2):
        chex.assert_rank(x1, 0)
        chex.assert_rank(x2, 0)

        return ((x1-x2)==0).astype(float)

class ScaleKernel(Kernel):
    def __init__(
        self,
        kernel: 'Kernel',
        variance: Optional[np.ndarray] = None,
    ) -> None:

        super(ScaleKernel, self).__init__(1, None)
        self.parent_kernel = kernel

        if variance is None:
            variance = 1.0
        else:
            ensure_float(variance)

        self.variance_param = Parameter(variance, constraint='positive', name='ScaleKernel/Variance')

    @property
    def variance(self) -> np.ndarray:
        return self.variance_param.value

    def K_diag(self, X1):
        return self.variance * self.parent_kernel.K_diag(X1)

    def _K(self, X1, X2):
        return self.variance * self.parent_kernel.K(X1, X2)

    def fix(self):
        self.variance_param.fix()
        self.parent_kernel.fix()

    def release(self):
        self.variance_param.release()
        self.parent_kernel.release()


class StationaryKernel(Kernel):
    def __init__(
        self,
        lengthscales: Optional[np.ndarray] = None,
        input_dim: Optional[int] = 1,
        active_dims: Optional[np.ndarray] = None,
        additive = False,
        name = None
    ) -> None:
        if lengthscales is None and input_dim is None:
            raise RuntimeError('Input dim must be passed')

        if input_dim is None:
            input_dim = len(lengthscales)

        super(StationaryKernel, self).__init__(input_dim, active_dims)

           # register lengthscales and variances
        if name is None:
            name = 'Kernel'

        self.base_name = name

        if active_dims is None:
            name = f'{name}/Lengthscale({additive})'
        else:
            name = f'{name}/Lengthscale({additive})[{active_dims}]'

        if type(lengthscales).__name__ == 'Parameter':
            # lengthscale is already a param so no need to create a new one
            self.lengthscale_param = lengthscales
        else:

            if lengthscales is None:
                lengthscales = np.array([1.0] * input_dim)
            else:
                lengthscales = ensure_array(lengthscales)

            chex.assert_shape(lengthscales, [input_dim])

            self.lengthscale_param = Parameter(lengthscales, constraint='positive', name=name)

        self.additive = additive

    def fix(self):
        self.lengthscale_param.fix()

    def release(self):
        self.lengthscale_param.release()

    @property
    def lengthscales(self) -> np.ndarray:
        return self.lengthscale_param.value

    def K_diag(self, X1):
        #TODO: this needs to be multiplied by D, or var is only used once!
        if self.additive:
            return np.ones(X1.shape[0]) * self.input_dim
        else:
            return np.ones(X1.shape[0])

    def _K(self, X1, X2):
        D = X1.shape[1]

        # TODO: do we want to jits here?

        #@partial(jit, static_argnums=(3))
        def _K_d2(x1, x2, lengthscales, additive):
            #vectorised over 2nd input
            chex.assert_rank(x1, 1)
            chex.assert_rank(x2, 1)

            chex.assert_equal(x1.shape[0], D)
            chex.assert_equal(x2.shape[0], D)

            k_d1_d2 = jax.vmap(self._K_scaler, in_axes=[0, 0, 0])(x1, x2, lengthscales)

            chex.assert_equal(k_d1_d2.shape[0], D)

            if additive:
                k_xx =  np.sum(k_d1_d2)
            else:
                k_xx =  np.prod(k_d1_d2)

            chex.assert_rank(k_xx, 0)

            return k_xx


        #@partial(jit, static_argnums=(3))
        def _K_d1(x1, X2, lengthscales, additive):
            #vectorised over first input
            return jax.vmap(_K_d2, in_axes=[None, 0, None, None], out_axes=0)(x1, X2, lengthscales, additive)

        K = jax.vmap(_K_d1, in_axes=[0, None, None, None], out_axes=0)(X1, X2, self.lengthscales, self.additive)

        chex.assert_equal(K.shape[0], X1.shape[0])
        chex.assert_equal(K.shape[1], X2.shape[0])

        return K

class StationaryVarianceKernel(StationaryKernel):
    def __init__(
        self,
        lengthscales: Optional[np.ndarray] = None,
        variance: Optional[np.ndarray] = None,
        input_dim: Optional[int] = 1,
        active_dims: Optional[np.ndarray] = None,
    ) -> None:

        super(StationaryVarianceKernel, self).__init__(lengthscales, input_dim, active_dims)

        # register lengthscales and variances
        self.variance_param = Parameter(variance, constraint='positive')

    @property
    def variance(self) -> np.ndarray:
        return self.variance_param.value

    def _K_scaler(self, x1, x2, lengthscale):
        return self._K_scaler_with_var(x1, x2, lengthscale, self.variance)

    def K_diag(self, X1):
        return self.variance * np.ones(X1.shape[0])

    def fix(self):
        self.lengthscale_param.fix()
        self.variance_param.fix()

    def release(self):
        self.lengthscale_param.release()
        self.variance_param.release()

class NonStationaryKernel(Kernel):
    def __init__(self) -> None:
        super(Kernel, self).__init__()

        pass

class Linear(Kernel):
    def __init__(
        self,
        offset: Optional[np.ndarray] = None,
    ) -> None:

        super(Linear, self).__init__(1, None)
        if offset is None:
            variance = 0.0
        else:
            ensure_float(offset)

        self.offset_param = Parameter(offset,  name='Linear/offset')

    def _K(self, X1, X2):
        c = self.offset_param.value
        return (X1-c) @ (X2-c).T

    def K_diag(self, X1):
        return np.square(self._apply_active_dim(X1))[:, 0]


