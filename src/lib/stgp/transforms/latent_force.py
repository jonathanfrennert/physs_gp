from .transform import Transform, LinearTransform, NonLinearTransform, Independent
from .sdes import SDE
from .. import Parameter
from ..computation.matrix_ops import to_block_diag
from ..computation.parameter_transforms import softplus

import jax
import jax.numpy as np
from jax import jacfwd

class LatentForce(SDE):
    pass

class LinearLFM(LatentForce):
    pass


class NonLinearLFM(LatentForce):
    def __init__(self, latent_lti_sde):

        self.latent_obj = latent_lti_sde

    def f(self, x, X_s, t):
        x_lfm = x[:self.state_size]
        x_sde = x[self.state_size:]

        base_f = self.latent_obj.f(x_sde, X_s, t)

        H = self.latent_obj.H(x, X_s, t)
        u = H @ x_sde

        x_lfm = np.vstack([x_lfm, u])

        lfm_f = self.f_lfm(x_lfm, X_s, t)

        return np.vstack([lfm_f, base_f])

    def L(self, x, X_s, t):
        L = self.latent_obj.L(x, X_s, t)
        L_lfm = np.eye(self.state_size)*0.0

        return to_block_diag([L_lfm, L])

    def Q(self, x, X_s, t):
        Q = self.latent_obj.Q(x, X_s, t)
        Q_lfm = np.eye(self.state_size)*0.0

        return to_block_diag([Q_lfm, Q])

    def H(self, x, X_s, t):
        # Only select the LFM state
        H = self.latent_obj.H(x, X_s, t)*0.0

        H_lfm = self.H_lfm(x, X_s, t)

        # pad with zeros to make shape compatable
        n_zero_to_add = H.shape[1] - H_lfm.shape[1]
        #H_lfm = np.pad(H_lfm, ((0, 0), (0, n_zero_to_add)))

        return np.hstack([H_lfm, H])

    def P_inf(self, x, X_s, t):
        lfm_P_inf = self.P_inf_lfm(x, X_s, t)
        base_P_inf = self.latent_obj.P_inf(x, X_s, t)

        return to_block_diag([lfm_P_inf, base_P_inf])

    def m_inf(self, x, X_s, t):
        lfm_m_inf = self.m_inf_lfm(x, X_s, t)
        base_m_inf = self.latent_obj.m_inf(x, X_s, t)

        return np.vstack([lfm_m_inf, base_m_inf])


class LotkaVolterra(NonLinearLFM):
    def __init__(self, latent, alpha, beta, delta, gamma, init_state = None):
        super(LotkaVolterra, self).__init__(latent)

        self.alpha = Parameter(np.array(alpha), name='alpha')
        self.beta = Parameter(np.array(beta), name='beta')
        self.delta = Parameter(np.array(delta), name='delta')
        self.gamma = Parameter(np.array(gamma), name='gamma')

        self.state_size = 2

        if init_state is None:
            init_state = 10.0*np.ones(self.state_size)[:, None]

        self.init_state = Parameter(
            np.reshape(np.array(init_state), [self.state_size, 1]),
            name = 'init_state' 
        )


    def _f(self, state_x, t):
        """ Evalulate dx/dt"""
        alpha = self.alpha.value
        beta = self.beta.value
        delta = self.delta.value
        gamma = self.gamma.value

        x = state_x[0]
        y = state_x[1]

        return np.array([
            alpha * x - beta * x * y,
            delta * x * y - gamma * y 
        ])



    def f_lfm(self, state_x, X_s, t):
        """ Evalulate dx/dt"""
        alpha = self.alpha.value
        beta = self.beta.value
        delta = self.delta.value
        gamma = self.gamma.value

        x = state_x[0]
        y = state_x[1]
        u1 = state_x[2]
        u2 = state_x[3]

        return np.array([
            alpha * x - beta * x * y + u1,
            delta * x * y - gamma * y + u2 
        ])

    def H_lfm(self, x, X_s, t):
        return np.eye(self.state_size)

    def L_lfm(self, x, X_s, t):
        return np.ones(self.state_size)[:, None]  


    def m_inf_lfm(self, x, X_s, t):
        return self.init_state.value

    def P_inf_lfm(self, x, X_s, t):
        return np.eye(self.state_size)

class RM_Population(NonLinearLFM):
    def __init__(self, latents, alpha, K, beta, b, gamma, delta, init_state = None):
        super(RM_Population, self).__init__(latents)

        self.alpha = Parameter(np.array(alpha, dtype=np.float64), constraint='positive', name='alpha')
        self.K = Parameter(np.array(K, dtype=np.float64), constraint='positive', name='K')
        self.beta = Parameter(np.array(beta, dtype=np.float64), constraint='positive', name='beta')
        self.b = Parameter(np.array(b, dtype=np.float64), constraint='positive', name='b')
        self.gamma = Parameter(np.array(gamma, dtype=np.float64), constraint='positive', name='gamma')
        self.delta = Parameter(np.array(delta, dtype=np.float64), constraint='positive', name='delta')

        self.state_size = 2

        if init_state is None:
            init_state = 10.0*np.ones(self.state_size)[:, None]

        self.init_state = Parameter(
            np.reshape(np.array(init_state), [self.state_size, 1]),
            name = 'init_state' 
        )

    def _f(self, state_x, t):
        """ Evalulate dx/dt"""
        alpha = self.alpha.value
        K = self.K.value
        beta = self.beta.value
        b = self.b.value
        gamma = self.gamma.value
        delta = self.delta.value

        x, y = state_x[0], state_x[1]

        return np.array([
            x * (alpha * (1-(x/K)) - beta * y / (b + x)),
            y * ( delta * x / (b+x) - gamma) 
        ])

    def f_lfm(self, state_x, X_s, t):
        """ Evalulate dx/dt"""
        x = state_x[0]
        y = state_x[1]
        u1 = state_x[2]
        u2 = state_x[3]

        ob_state = [x, y]

        d = self._f(ob_state, t)

        return np.array([
            d[0] + u1,
            d[1] + u2 
        ])

    def H_lfm(self, x, X_s, t):
        return np.eye(self.state_size)

    def L_lfm(self, x, X_s, t):
        return np.ones(self.state_size)[:, None]  


    def m_inf_lfm(self, x, X_s, t):
        return self.init_state.value

    def P_inf_lfm(self, x, X_s, t):
        return np.eye(self.state_size)


class PopulationLotkaVolterra(NonLinearLFM):
    """ Following https://jckantor.github.io/CBE30338/02.05-Hare-and-Lynx-Population-Dynamics.html """
    def __init__(self, latent, a, b, c, d, k, r, init_state = None):
        super(PopulationLotkaVolterra, self).__init__(latent)

        self.a = Parameter(np.array(a), name='a')
        self.b = Parameter(np.array(b), name='b')
        self.c = Parameter(np.array(c), name='c')
        self.d = Parameter(np.array(d), name='d')
        self.k = Parameter(np.array(k), name='k')
        self.r = Parameter(np.array(r), name='r')

        self.state_size = 2

        if init_state is None:
            init_state = 10.0*np.ones(self.state_size)[:, None]

        self.init_state = Parameter(
            np.reshape(np.array(init_state), [self.state_size, 1]),
            name = 'init_state',
            train=True
        )


    def _f(self, state_x, t):
        """ Evalulate dx/dt"""
        a = self.a.value
        b = self.b.value
        c = self.c.value
        d = self.d.value
        k = self.k.value
        r = self.r.value


        H = state_x[0]
        L = state_x[1]

        dH =  r*H*(1-H/k) - a*H*L/(c+H)
        dL = b*a*H*L/(c+H) - d*L

        return np.array([
            dH, dL  
        ])

    def f_lfm(self, state_x, X_s, t):
        """ Evalulate dx/dt"""
        a = self.a.value
        b = self.b.value
        c = self.c.value
        d = self.d.value
        k = self.k.value
        r = self.r.value

        H = state_x[0]
        L = state_x[1]

        u1 = state_x[2]
        u2 = state_x[3]

        dH =  r*H*(1-H/k) - a*H*L/(c+H) + u1
        dL = b*a*H*L/(c+H) - d*L + u2

        return np.array([
            dH, dL  
        ])

    def H_lfm(self, x, X_s, t):
        return np.eye(self.state_size)

    def L_lfm(self, x, X_s, t):
        return np.ones(self.state_size)[:, None]  


    def m_inf_lfm(self, x, X_s, t):
        return self.init_state.value

    def P_inf_lfm(self, x, X_s, t):
        return np.eye(self.state_size)

class Linearized(LinearLFM):
    def __init__(self, base_lfm, mean_state):
        self.base_lfm = base_lfm
        self.mean_state = mean_state

    def _f(self, init_x, step):

        # compute jacobian
        F_fn = lambda m: self.base_lfm._f(m, step)
        F = jacfwd(F_fn)(self.mean_state)

        #Linearisation step
        return  self.mean_state + F @ (self.mean_state - init_x)

class LinearODE(LinearLFM):
    def _f(self, state_x, t):
        """ Evalulate dx/dt"""
        x = state_x[0]
        y = state_x[1]

        return np.array([
            x-y,
            y-x 
        ])
