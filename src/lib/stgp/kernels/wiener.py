import objax
import chex
import jax
import jax.numpy as np
from jax.scipy.linalg import expm
from jax.scipy.special import factorial

from . import StationaryKernel, StationaryVarianceKernel, MarkovKernel
from .. import Parameter

def symmetric_abs(a, b):
    """ Force dabs(x)/dx at x=0 to be 0 """
    tau = np.abs(a-b)
    #return tau
    return tau * np.where(np.isinf(1/tau), 0.0, 1.0)

class Wiener(MarkovKernel):
    def __init__(self, q=1):
        super(Wiener, self).__init__()
        
        self.q_param = Parameter(q, constraint='positive', name='IntegratedWiener/q', train=False)

    def _K_scaler(self, x1, x2):
        """
        K(X1, X2) = theta*min(x1, x2)
        """

        q = self.q_param.value
        _min = np.min(np.array([x1, x2]))

        return q * _min

    def K_diag(self, X1):
        q = self.q_param.value
        return np.squeeze(X1) * q

def _integrated_wiener_coef(q, x1, x2):
    _min = np.min(np.array([x1, x2]))
    _max = np.max(np.array([x1, x2]))

    if q == 0:
        ai = 1.0
        bi = 0.0
        ri = 0.0
    elif q == 1:
        ai = 3
        bi = 1/2.0
        ri = 1
    elif q == 2:
        ai = 20
        bi = 1/12.0
        ri = (x1+x2)-0.5*_min
    elif q == 3:
        ai = 252
        bi = 1/720.0
        ri = 5*(_max**2)+2*x1*x2+3*(_min**2)

    return ai, bi, ri

class WienerVelocity(MarkovKernel):
    """
    Following https://github.com/alshedivat/gpml/blob/master/cov/covW.m
    """
    def __init__(self, q=1, variance=1.0, stable_state_covariance=0.0, m_init = None, train_m_init=False, active_dims = None):
        super(WienerVelocity, self).__init__()
        
        self.q = int(q)
        self.variance_param = Parameter(variance, constraint='positive', name=f'IntegratedWiener({self.q})/variance')
        self._state_space_dim = q+1
        self.explicit_q = True

        if m_init is None:
            m_init = np.zeros(self.state_size())[:, None]
        else:
            m_init = np.reshape(np.array(m_init), [self.state_size(), 1])

        self.m_init_param = Parameter(m_init, name=f'IntegratedWiener({self.q})/m_init', train=train_m_init)
        self.stable_state_covariance = stable_state_covariance

        self.input_dim = 1
        self.active_dims = active_dims

    @property
    def m_init(self):
        return self.m_init_param.value
        
    def state_size(self):
        return self._state_space_dim

    def to_ss(self, X_spatial=None):
        q = self.q
        var = self.variance_param.value

        dim = self.state_size()

        F = np.eye(dim, k = 1)
        L = np.hstack([np.zeros(dim-1), [1]])[:, None]
        H = np.hstack([[1], np.zeros(q)])[None, :]
        Pinf = np.eye(self.state_size())*self.stable_state_covariance # low variance on x, high variance on the derivatives
        Qc = var
        minf = self.m_init

        return F, L, Qc, H, minf, Pinf

    def expm(self, dt, X_spatial=None):
        #F, _, _, _, _ = self.to_ss(X_spatial) 
        #_expm =  expm( F * dt)

        dim = self.state_size()
        idx = np.arange(dim)
        _weights = jax.vmap(
            lambda i:
                jax.vmap(
                    lambda j:
                        (dt**(j-i))/(factorial(j-i))
                        
                )(idx)
        )(idx)

        #mask out lower triange inf values
        _expm = np.nan_to_num((~np.isinf(_weights))*_weights)
        return _expm

    def Q(self, dt, A, P_inf, X_spatial=None):
        dim = self.state_size()
        q = self.q

        idx = np.arange(dim)

        #dt = 0.1
        Q = jax.vmap(
            lambda i:
                jax.vmap(
                    lambda j:
                        (
                            dt**(2*q+1-i-j)
                        )/(
                            (2*q+1-i-j)*
                            factorial(q-i) * 
                            factorial(q-j)
                        )
                        
                )(idx)
        )(idx)

        #Q_true = np.array([[1/300.0, 1/20.0], [1/20.0, 1.0]])/10
        #breakpoint()

        return Q * (self.variance_param.value)


    def K_diag(self, X1):
        q = self.q
        var = self.variance_param.value

        ai, bi, ri = _integrated_wiener_coef(q, X1[0], X1[0])

        return var * (1/ai)*(np.squeeze(X1)**(2*q+1))

    def _K_scaler(self, x1, x2):
        """
        if q ==1:
            K(X1, X2) = theta^2((1.0/3.0) * min(x1, x2)^3 + |a-b|*0.5*min(a, b))

        when computing derivatives through this kernel, all derivates are wrt to x1
        """

        q = self.q
        var = self.variance_param.value

        tau = symmetric_abs(x1, x2)
        #_min = symmetric_min(x1, x2)
        _min = np.min(np.array([x1, x2]))
        ai, bi, ri = _integrated_wiener_coef(q, x1, x2)

        _k = (1/ai)*_min**(2*q+1) + bi*_min**(q+1) * tau * ri
        return var * _k

IntegratedWiener = WienerVelocity

