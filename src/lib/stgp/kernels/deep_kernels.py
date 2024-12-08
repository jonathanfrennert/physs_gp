from . import Kernel
from .kernel import StationaryKernel, ConcatationKernel, WhiteNoiseKernel
from ..dispatch import evoke
from .. import settings
from ..computation.gaussian import log_gaussian_scalar
import objax
from .. import Parameter

import jax
import jax.numpy as np

from jax.scipy.special import erf

from typing import List, Optional, Union

import chex 
import warnings

class DeepIndependentKernel(ConcatationKernel):
    """ TODO: check if still needeed"""
    def K(self, X1: np.array, X2: np.array):
        K_arr = super(DeepIndependentKernel, self).K(X1, X2)

        return np.sum(K_arr, axis=0)

    def K_diag(self, X1: np.array):
        K_arr = super(DeepIndependentKernel, self).K_diag(X1)
        return np.sum(K_arr, axis=0)

class DeepKernel(Kernel):
    def __init__(
        self,
        parent: Optional['Model'] = None,
        kernel: Optional['Kernel'] = None,
    ):
        # setup active dims and input_dim
        super(DeepKernel, self).__init__()

        # If parent is not passed in the kernel constructed it MUST be added before the kernel is used
        self.parent = parent

        self.kernel = kernel

        if self.kernel is None:
            self.use_parent = True
        else:
            self.use_parent = False

    def set_parent(self, parent):
        self.parent = parent

    def forward(self, X1, X2, mu_1, mu_2, k_x1, k_x2, K_x1x2):
        mu_1 = mu_1[None, :, :]
        mu_2 = mu_2[None, :, :]
        k_x1 = k_x1[None, :, :]
        k_x2 = k_x2[None, :, :]
        K_x1x2 = K_x1x2[None, :, :]

        return self._K_with_pm(
            X1, X2, mu_1, mu_2, k_x1, K_x1x2, k_x2
        )
        return K

    def propogate_parent_var(self, x1):
        if self.use_parent:
            pm_x1 = self.parent.mean_blocks(x1)
            pk_x1x1 = self.parent.var_blocks(x1)
            chex.assert_rank([pm_x1, pk_x1x1], [3, 3])

        else:
            N1 = x1.shape[0]
            N2 = x2.shape[0]
            pm_x1 = np.zeros([1, N1, 1])

            pk_x1x1 = self.kernel.K_diag(x1)[None, :, None]


        vec_shape_1 = [self.input_dim, x1.shape[0], 1]

        return np.reshape(pm_x1, vec_shape_1),  np.reshape(pk_x1x1, vec_shape_1)

    def propogate_parent(self, x1, x2):
        if self.use_parent:
            pm_x1 = self.parent.mean_blocks(x1)
            pm_x2 = self.parent.mean_blocks(x2)

            pk_x1x1 = self.parent.var_blocks(x1)
            pk_x2x2 = self.parent.var_blocks(x2)
            pk_x1x2 = self.parent.covar_blocks(x1, x2)

            chex.assert_rank([pm_x1, pm_x2], [3, 3])
            chex.assert_rank([pk_x1x1, pk_x2x2, pk_x1x2], [3, 3, 3])

        else:
            N1 = x1.shape[0]
            N2 = x2.shape[0]
            pm_x1 = np.zeros([1, N1, 1])
            pm_x2 = np.zeros([1, N2, 1])

            pk_x1x1 = self.kernel.K_diag(x1)[None, :, None]
            pk_x2x2 = self.kernel.K_diag(x2)[None, :, None]
            pk_x1x2 = self.kernel.K(x1, x2)[None, :, :]

            raise RuntimeError()


        vec_shape_1 = [self.input_dim, x1.shape[0], 1]
        vec_shape_2 = [self.input_dim, x2.shape[0], 1]

        return np.reshape(pm_x1, vec_shape_1) , np.reshape(pm_x2, vec_shape_2), np.reshape(pk_x1x1, vec_shape_1), np.reshape(pk_x2x2, vec_shape_2), pk_x1x2

    def _K(self, X1, X2):
        # Precompute parent mean and variances
        pm_x1, pm_x2, pk_x1x1, pk_x2x2, pk_x1x2 = self.propogate_parent(X1, X2)

        return self._K_with_pm(X1, X2,pm_x1,pm_x2,pk_x1x1,pk_x1x2, pk_x2x2)

    def _K_with_pm(self, X1, X2,pm_x1,pm_x2,pk_x1, pk_x1x2, pk_x2):
        raise NotImplementedError()

    def K_diag(self, X1):
        raise NotImplementedError()

class DeepUIKernel(DeepKernel):
    """
    Deep Kernel with Uncertain Inputs
    """
    def __init__(self, kernel, surrogate_model, noise = 1e-5):
        """
        Args:
            kernel: prior kernel
            surrogate_model: model used to get derivatives of the GP prior (with the same kernel)
        """
        # setup active dims and input_dim
        # call parent of deepkernel as we dont need all the setup of DeepKernel
        super(DeepKernel, self).__init__()

        self.parent_kernel = kernel
        self.surrogate_model = surrogate_model

        self.noise_param = Parameter(np.array(noise), constraint='positive', name='DeepUIKernel/noise')

    @property
    def noise(self) -> np.ndarray:
        return self.noise_param.value

    def _K(self, X1, X2):
        chex.assert_rank([X1, X2], [2, 2])

        K_base = self.parent_kernel.K(X1, X2)

        noise = np.reshape(self.noise, [1, 1])

        pred_x1, var_x1 = self.surrogate_model.predict_f(X1)
        pred_x2, var_x2 = self.surrogate_model.predict_f(X2)

        pred, var = self.surrogate_model.predict_f(np.vstack([X1, X2]), diagonal=False)
        # var is in latent-data format?
        N1 = X1.shape[0]
        N2 = X2.shape[0]
        # get (co-)variance of derivative
        covar = var[N1:N1*2, :][:, N1*2+N2:]

        # extract first derivative
        pred_x1 = pred_x1[:, 1][:, None]
        pred_x2 = pred_x2[:, 1][:, None]

        # extract first derivative variance
        var_x1 = var_x1[:, 1][:, None]
        var_x2 = var_x2[:, 1][:, None]

        if X1.shape[0] == X2.shape[0]:
            # white noise setting
            K_taylor = pred_x1 @ noise @ pred_x2.T
     
            return K_base + K_taylor  
        else:
            return K_base 

    def K_diag(self, X1):
        # TODO: very inefficient
        return np.diag(self._K(X1, X1))

class DeepLinear(DeepKernel):
    def _K_with_pm(self, X1, X2,pm_x1,pm_x2,pk_x1, pk_x1x2, pk_x2):
        # TODO: assume single latent function
        return pm_x1[0] @ pm_x2[0].T + pk_x1x2[0]

    def K_diag(self, X1):
        pm_x1, pk_x1x1 = self.propogate_parent_var(X1)

        return np.squeeze((pm_x1[0] * pm_x1[0] + pk_x1x1[0]))

class DeepStationary(StationaryKernel, DeepKernel):
    """
    Parent class of all Deep stationary kernels.
    All these kernels assume a single input dimension 
        - i.e these do not construct ARD kernels, this must be done explictly in the model construction
    """
    def __init__(
        self, 
        parent: Optional['Model'] = None,
        kernel: Optional['Kernel'] = None,
        lengthscale: Optional[np.ndarray] = None, 
        input_dim: Optional[int] = 1, 
        active_dims: Optional[np.ndarray] = None
    ):

        super(DeepStationary, self).__init__(lengthscale, input_dim, active_dims)
        
        # If parent is not passed in the kernel constructed it MUST be added before the kernel is used
        self.parent = parent

        self.kernel = kernel

        if self.kernel is None:
            self.use_parent = True
        else:
            self.use_parent = False

    def set_parent(self, parent):
        self.parent = parent

    def forward(self, X1, X2, mu_1, mu_2, k_x1, k_x2, K_x1x2):
        mu_1 = mu_1[None, :, :]
        mu_2 = mu_2[None, :, :]
        k_x1 = k_x1[None, :, :]
        k_x2 = k_x2[None, :, :]
        K_x1x2 = K_x1x2[None, :, :]

        return self._K_with_pm(
            X1, X2, mu_1, mu_2, k_x1, K_x1x2, k_x2
        )
        return K

    def forward_diag(self, X1, mu_1, K_diag):
        return self.K_diag(X1)

    def K_diag(self, X1):
        return self._K_var(self.lengthscales) * np.ones(X1.shape[0])

    def _K(self, X1, X2):
        # Precompute parent mean and variances
        pm_x1, pm_x2, pk_x1x1, pk_x2x2, pk_x1x2 = self.propogate_parent(X1, X2)

        return self._K_with_pm(X1, X2,pm_x1,pm_x2,pk_x1x1,pk_x1x2, pk_x2x2)

    def _K_with_pm(self, X1, X2,pm_x1,pm_x2,pk_x1, pk_x1x2, pk_x2):
        D = X1.shape[1]

        N1 = X1.shape[0]
        N2 = X2.shape[0]

        chex.assert_shape(X1, [N1, D])
        chex.assert_shape(X2, [N2, D])
        chex.assert_shape(pm_x1, [self.input_dim, N1, 1])
        chex.assert_shape(pm_x2, [self.input_dim, N2, 1])
        chex.assert_shape(pk_x1, [self.input_dim, N1, 1])
        chex.assert_shape(pk_x2, [self.input_dim, N2, 1])
        chex.assert_shape(pk_x1x2, [self.input_dim, N1, N2])

        # Remove uncessary extra dim. Also ensures k_scalar returns a kernel not a matrix.
        pm_x1 = pm_x1[..., 0]
        pm_x2 = pm_x2[..., 0]
        pk_x1 = pk_x1[..., 0]
        pk_x2 = pk_x2[..., 0]

        # Batch over X1, and X2 to compute full K(X1, X2)
        def _K_d2(x1, x2, pm_x1, pm_x2, pk_x1, pk_x2, pk_x1x2):
            """ Computes K(x1, x2) """
            chex.assert_shape(x1, [D])
            chex.assert_shape(x2, [D])

            # batch over input_dim = first dimension 
            k_xx = jax.vmap(
                self._K_scaler,
                in_axes = [None, None, 0, 0, 0, 0, 0, 0],
                out_axes=0
            )(x1, x2, self.lengthscales, pm_x1, pm_x2, pk_x1, pk_x2, pk_x1x2)

            k_xx = np.prod(k_xx)

            return k_xx

        def _K_d1(x1, X2, pm_x1, pm_x2, pk_x1, pk_x2, pk_x1x2):
            """ Computes K(x1, X2) """
            return jax.vmap(
                _K_d2, 
                in_axes=[None, 0, None, 1, None, 1, 1],
                out_axes=0
            )(x1, X2, pm_x1, pm_x2,  pk_x1, pk_x2, pk_x1x2)

        K = jax.vmap(
            _K_d1, 
            in_axes=[0, None, 1, None, 1, None, 1], 
            out_axes=0
        )(X1, X2, pm_x1, pm_x2, pk_x1, pk_x2, pk_x1x2)

        chex.assert_shape(K, [X1.shape[0], X2.shape[0]])

        return K

class DeepRBF(DeepStationary):
    def _K_var(self, lengthscale):
        #return np.sqrt(np.pi*lengthscale)
        #return np.sqrt(2 * np.pi*lengthscale)/np.sqrt(2*np.pi*(lengthscale))
        return 1.0

    def _K_scaler(self, x1, x2, lengthscale, m1, m2, k_11, k_22, k_12):
        chex.assert_rank(k_11, 0)
        chex.assert_rank(k_22, 0)
        chex.assert_rank(k_12, 0)
        chex.assert_rank(lengthscale, 0)

        if self.use_parent:
            L = lengthscale + k_11 + k_22 - 2*k_12

            const = np.squeeze(np.sqrt(2 * np.pi*lengthscale))

            # setup correct dimensions for log_gaussian_scalar
            k_ij = const * np.exp(log_gaussian_scalar(0, m1-m2, L))
        else:
            raise RuntimeError()
            L = lengthscale + k_11 + k_22 - 2*k_12
            k_ij =  np.sqrt(lengthscale) / np.sqrt(L)

        chex.assert_rank(k_ij, 0)
        return k_ij

        
class DeepMatern12(DeepStationary):
    def __init__(
        self, 
        kernel: Optional['Kernel'] = None,
        parent_model: Optional['Model'] = None,
        lengthscale: Optional[np.ndarray] = None, 
        variance: Optional[np.ndarray] = None, 
        input_dim: Optional[int] = 1, 
        active_dims: Optional[np.ndarray] = None
    ):

        super(DeepMatern12, self).__init__(lengthscale, variance, input_dim, active_dims)

        self.parent_kernel = kernel
        self.parent_model = parent_model


    def _K_scaler(self, x1, x2, variance, lengthscale):
        raise NotImplementedError()
        #TODO: generalise to multi dimensions

        parent_mean, parent_k = self.propogate_parent(x1, x2)

        chex.assert_rank(parent_k, 2)

        k_11 = parent_k[0, 0]
        k_12 = parent_k[0, 1]
        k_22 = parent_k[1, 1]
        
        if parent_mean is None:
            raise NotImplementedError()
        else:
            m = np.abs(parent_mean[0] - parent_mean[1])
            k = k_11 + k_22 - 2*k_12

            a_1 = (-k/lengthscale)+m
            a_2 = (k/lengthscale)+m

            def component(a, neg=False):
                k_sqrt = np.sqrt(2*np.clip(k, settings.jitter))
                #k_sqrt = np.sqrt(2*k)
                
                b = (- a)/k_sqrt

                c_1 = variance 
                c_2 = -(1/k_sqrt)*(m**2 - a**2)

                if neg is False:
                    return  c_1* 0.5*(1+ erf(b))*np.exp(c_2)
                else: 
                    return  c_1* 0.5*( 1- erf(b))*np.exp(c_2)

            c_1 = component(a_1, neg=True)
            c_2 = component(a_2, neg=False)

            Kij = c_1 + c_2

        return Kij

        #return variance * (1/np.sqrt(1 + (k_11 + k_22 - 2*k_12)/(lengthscale)))

class DeepSMComponent(DeepStationary):
    def __init__(
        self, 
        kernel: Optional['Kernel'] = None,
        parent_model: Optional['Model'] = None,
        lengthscale: Optional[np.ndarray] = None, 
        variance: Optional[np.ndarray] = None, 
        input_dim: Optional[int] = 1, 
        active_dims: Optional[np.ndarray] = None
    ):
        super(DeepSMComponent, self).__init__(lengthscale, variance, input_dim, active_dims)

        self.parent_kernel = kernel
        self.parent_model = parent_model

    def lengthscales(self, raw_getter) -> np.ndarray:
        """ \mu is not constrained to be positive in the SM kernel. """
        return raw_getter()

    def _K_scaler(self, x1, x2, variance, lengthscale):
        raise NotImplementedError()
        #TODO: generalise to multi dimensions

        parent_mean, parent_k = self.propogate_parent(x1, x2)

        chex.assert_rank(parent_k, 2)

        k_11 = parent_k[0, 0]
        k_12 = parent_k[0, 1]
        k_22 = parent_k[1, 1]
        
        if parent_mean is None:
            raise NotImplementedError()
        else:
            m = np.abs(parent_mean[0] - parent_mean[1])
            k = k_11 + k_22 - 2*k_12
            k = np.clip(k, settings.jitter)

            v = variance
            mu = lengthscale

            lam = 1/(4*np.pi*np.pi*v)

            b = (1/((1/k)+(1/lam)))
            a = (1/b) * ((1/k)*m)

            c = np.sqrt(2*np.pi*variance)

            gauss = lambda x, m, v : (1/np.sqrt(2*np.pi*v))*np.exp(-(1/(2*v))*(x-m)**2)

            print('exp(): ', np.exp((-1/4)*(2*np.pi*mu)**2))
            print('np.sqrt(np.pi/(2*b)): ', np.sqrt(np.pi/(2*b)))

            Kij = np.sqrt(2*np.pi*lam)*gauss(m, 0, k+lam) * np.exp((-1/2)*(b*(2*np.pi*mu))**2)
            
            #Kij = np.sqrt(2*np.pi*lam)*np.cos(2*np.pi*mu*a)*np.exp(-(1/(2*b))*(2*np.pi*mu)**2)*gauss(m, 0, k+lam)

        return Kij

