from . import Transform, Independent
from ..computation.permutations import data_order_to_output_order, ld_to_dl, dl_to_ld
from ..computation.matrix_ops import to_block_diag
from .. import Parameter

import jax.numpy as np

class SDE(Transform):

    @property
    def latents(self):
        return self.gp.latents

    @property
    def num_latents(self):
        return len(self.latents)

class LTI_SDE(SDE):
    """ @TODO: Only for single latent functions"""
    def __init__(self, gp: 'Model', m_init=None, train_m_init = True):
        self.gp = gp
        self.whiten_space = False # required for api consistentcy
        self._parent = self.gp

        if m_init is not None:
            # ensure correct dimension
            m_init = np.reshape(np.array(m_init), [-1, 1])
            self.m_init_param = Parameter(m_init, name=f'LTI_SDE/m_init', train=train_m_init)
        else:
            self.m_init_param = None

    @property
    def temporal_output_dim(self):
        """
        Only return `f'
        """
        return 1

    @property
    def _output_dim(self):
        #return 1
        return self.gp.output_dim

    @property
    def spatial_output_dim(self):
        # TODO: this is a hacky atm

        # if the spatial kernel is a derivate kernel this will get ignored by the filter and computed explicitely after smoothing.
        # i.e when in a hierachical model the filter does not compute sptial derivates and so the spatial dim here will be 1
        try:
            return self.gp.base_prior.parent[0].kernel.k2.output_dim
        except Exception as e:
            return 1

    def state_space_dim(self):
        return self.gp.state_space_dim()

    def state_space_representation(self, X_s, dt, t):
        return self.gp.state_space_representation(X_s)

    def f(self, x, X_s, t):
        F, _, _, _, _, _ = self.gp.state_space_representation(X_s)

        return F @ x

    def L(self, x, X_s, t):
        _, L, _, _, _, _ = self.gp.state_space_representation(X_s)

        return L

    def Qc(self, x, X_s, t):
        _, _, Q, _, _, _ = self.gp.state_space_representation(X_s)

        return Q

    def H(self, x, X_s, t):
        _, _, _, H, _, _ = self.gp.state_space_representation(X_s)

        return H

    def P_inf(self, x, X_s, t):
        _, _, _, _, _, Pinf = self.gp.state_space_representation(X_s)

        return Pinf

    def m_inf(self, x, X_s, t):
        if self.m_init_param is None:
            _, _, _, _, m_inf, _ = self.gp.state_space_representation(X_s)
        else:
            return self.m_init_param.value
        return m_inf

    def expm(self, X_s, t):
        return self.gp.expm(t, X_s)

    def Q(self, dt_k, A_k, P_inf, X_spatial=None):
        return self.gp.Q(dt_k, A_k, P_inf, X_spatial=X_spatial)

class LTI_SDE_Full_State_Obs(LTI_SDE):
    """ For consistentcy with LTI_SDE all dimensions correspond to a single latent function """
    def __init__(self, gp: 'Model', whiten_space=False, overwrite_H=True, keep_dims = None, permute=True, m_init=None, train_m_init = True):
        super(LTI_SDE_Full_State_Obs, self).__init__(gp, m_init=m_init, train_m_init = train_m_init)
        self.gp = gp
        self._state_space_dim = sum(self.gp.state_space_dim())
        self.whiten_space = whiten_space
        self._num_latents = len(self.gp.state_space_dim())

        # select all dims per latent functino
        if keep_dims is None:
            self.keep_dims = np.array(range(self.gp.state_space_dim()[0]))
        else:
            self.keep_dims = np.array(keep_dims)

        self.overwrite_H = overwrite_H
        self.permute = permute
        self.m_init_param = None



    @property
    def temporal_output_dim(self):
        """ Returns the full state.  """
        return self._state_space_dim

    @property
    def _output_dim(self):
        return  self.temporal_output_dim * self.spatial_output_dim

    def H(self, x, X_s, t):

        Q = len(self.gp.state_space_dim())

        if self.overwrite_H:
            # for most kernels we can just set H to eye to observe f and its (time) derivatives
            # Observe both f and df
            H_t = np.eye(self._state_space_dim)
        else:
            # for some kernels, like the periodic kernel) the state does not correspond exactly
            #   to f and its time derivatives, so this needs to be handled by the kernel itself
            _, _, _, H_t, _, _ = self.gp.state_space_representation(X_s)
            _H_t = H_t # for debugging

            if not self.permute:
                return H_t

        # need to permute from latent-ds-space-df to latent-df-ds-space
        if X_s is None:
            Ns = 1
        else:
            Ns = X_s.shape[0]

        dt = self.gp.state_space_dim()[0]
        ds = self.spatial_output_dim
        dt_keep = len(self.keep_dims)

        full_dim = dt * ds * Ns
        mask_dim = dt_keep * ds * Ns
        I = np.eye(full_dim)
        I = np.reshape(I, [ds * Ns, dt, full_dim])[:, self.keep_dims, :]
        I_mask = np.reshape(I, [mask_dim, full_dim])

        # convert from [ds, ns, dt] -> [dt_keep, ds, ns]
        H_q = ld_to_dl(num_latents = ds * Ns, num_data = dt_keep) @ I_mask

        # convert from [Q, ds, ns, dt] -> [Q, dt_keep, ds, ns]
        H = to_block_diag([
            H_q
            for q in range(Q)
        ])


        return H

class LTI_SDE_Full_State_Obs_With_Mask(LTI_SDE_Full_State_Obs):
    """
    Observe partial deriatives. Useful when we have observations on [f, df/dt] but we want to use a smoother kernel like the matern52/72 etc.
    """
    def __init__(self, gp: 'Model', keep_dims, whiten_space=False, overwrite_H=True, permute=True):
        self.gp = gp
        self._state_space_dim = sum(self.gp.state_space_dim())
        self.keep_dims = np.array(keep_dims)
        self.whiten_space = whiten_space,
        self.overwrite_H = overwrite_H
        self.permute=permute
        self.m_init_param = None

    @property
    def temporal_output_dim(self):
        """ Returns the full state.  """
        return self.keep_dims.shape[0]*self.num_latents

class EulerMaruyama(SDE):
    def __init__(self, base_sde):
        self.base_sde = base_sde
        self.whiten_space = False # required for api consistentcy

    def H(self, x, X_s, t):
        return self.base_sde.H(x, X_s, t)

    def P_inf(self, x, X_s, t):
        return self.base_sde.P_inf(x, X_s, t)

    def m_inf(self, x, X_s, t):
        return self.base_sde.m_inf(x, X_s, t)

    def f_dt(self, x, X_s, t, dt):
        base_f = self.base_sde.f(x, X_s, t)
        return x + base_f * dt

    def Sigma_dt(self, x, X_s, t, dt):
        L = self.base_sde.L(x, X_s, t)
        Q = self.base_sde.Q(x, X_s, t)

        return L @ Q @ L.T * dt

    def _f(self, x, t):
        return self.base_sde._f(x, t)


class LinearizedFilter_SDE(LTI_SDE):

    @property
    def _output_dim(self):
        return 4

