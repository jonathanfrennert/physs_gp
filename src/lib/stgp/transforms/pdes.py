import jax
import jax.numpy as np
import chex

from ..core import Block 
from . import Transform, LinearTransform, Joint, Independent

from ..dispatch import evoke
from ..computation.matrix_ops import to_block_diag, get_block_diagonal
from .. import Parameter
from ..computation.matrix_ops import hessian
from jax import jacfwd, jacrev, grad
import objax
import numpy as onp

class DifferentialOperatorJoint(LinearTransform, Joint):
    """
    tbd.
    """
    def __init__(
        self,
        base_latent,
        mean = None,
        kernel = None,
        is_base:bool = True,
        has_parent:bool = False,
        hierarchical = False,
        whiten_space=False
    ):
        if base_latent is None:
            raise RuntimeError('Latent gp must be passed')

        self._parent = base_latent
        self.derivative_mean = mean
        self.derivative_kernel = kernel
        self._output_dim = self.derivative_kernel.output_dim
        self._input_dim = 1
        self._is_base = is_base
        self.has_parent = has_parent
        self.hierarchical = hierarchical
        self.whiten_space = whiten_space

    @property
    def full_transform(self):
        # do not use batched transform
        return True

    @property
    def is_base(self):
        return self._is_base

    @property
    def out_block_dim(self):
        return self.derivative_kernel.output_dim

    @property
    def in_block_dim(self):
        # always requires the full input covariance to compute derivates
        return Block.FULL

    @property
    def in_block_type(self):
        # always requires the full input covariance to compute derivates
        return Block.FULL

    def forward(self, f): return f

    def transform(self, mu, var, data):
        if not self.hierarchical:
            # base prior so no need to transform
            chex.assert_rank([mu, var], [2, 2])
        else:
            # compute 
            # no time
            chex.assert_rank([mu, var], [3, 4])
            mu, var = evoke('spatial_conditional', data, self)(
                data, 
                self.parent.parent.sparsity.raw_Z, 
                mu, 
                var[:, 0, ...], 
                self
            )

            chex.assert_rank([mu, var], [3, 4])

        return mu, var

    def get_sparsity_list(self):
        return [self.get_sparsity()]

    def get_Z_stacked(self):
        Z = self.get_Z_blocks()
        Z = Z[None, ...]
        chex.assert_rank(Z, 4)
        return Z

    def get_Z_blocks(self):
        Z = self.parent.get_Z()
        Z_arr = np.tile(Z, [self.output_dim, 1, 1])

        chex.assert_rank(Z_arr, 3)
        return Z_arr

    def get_Z(self):
        Z_all = self.get_Z_blocks()

        Z = np.vstack(Z_all)
        chex.assert_rank(Z, 2)
        return Z

    def get_sparsity(self):
        if self.has_parent:
            return self.parent.get_sparsity()
        else:
            return self.parent.sparsity

    def mean(self, X1):
        if not self.has_parent:
            return np.zeros([X1.shape[0] * self.output_dim, 1])
        else:
            mean_fn = self.parent.mean_blocks
            mean_x = self.derivative_mean.mean_from_fn(X1, mean_fn)
            return mean_x

    def mean_blocks(self, X1):
        if not self.has_parent:
            return np.zeros([self.output_dim, X1.shape[0], 1])
        else:
            mean_fn = self.parent.mean_blocks
            mean_x = self.derivative_mean.mean_blocks_from_fn(X1, mean_fn)
            return mean_x

    def b_mean(self, X1):
        # for compatability with Independent X1 and can either be of rank 3 or 4
        if len(X1.shape) == 4 :
            chex.assert_equal([X1.shape[0]], [1])
            X1 = X1[0]

        # We only support zero mean so we simply return the mean
        return self.mean(X1[0])

    def b_mean_blocks(self, X1):
        return self.mean_blocks(X1[0])

    def covar_from_fn(self, X1, X2, var_fn):
        K_xz = self.derivative_kernel.K_from_fn(X1, X2, var_fn)
        return K_xz

    def covar(self, X1, X2):
        # if self does not have a parent then 
        #  we know that the prior covariance is directly given by K
        #  so we can efficient computed the derivate kernel
        if not self.has_parent:
            return self.derivative_kernel.K(X1, X2)
        else:
            var_fn = self.parent.covar

            return self.covar_from_fn(X1, X2, var_fn)

    def b_covar(self, X1, X2):
        """ WARNING: we assume that X1, X2 is actually repeated """

        # for compatability with Independent X1 and can either be of rank 3 or 4
        if len(X1.shape) == 4 and  len(X2.shape) == 4:
            chex.assert_equal([X1.shape[0], X2.shape[0]], [1, 1])
            X1 = X1[0]
            X2 = X2[0]

        return self.covar(X1[0], X2[0])

    def covar_blocks(self, X1, X2):
        # TODO: this might need to be permutated 
        #raise NotImplementedError() 
        K_full = self.covar(X1, X2)

        return get_block_diagonal(
            K_full,
            self.output_dim,
        )

    def b_covar_blocks(self, X1, X2):
        """ WARNING: we assume that X1 is actually repeated """

        return self.covar_blocks(X1[0], X2[0])

    def full_var(self, X):
        return self.covar(X, X)

    def var(self, X):
        fn = jax.vmap(lambda p,x: p.covar(x[None, :], x[None, :]), [None, 0])
        res = fn(self, X)
        diag_vec = np.diagonal(res, axis1=1, axis2=2)
        res =  np.hstack(diag_vec.T)[:, None]

        chex.assert_rank(res, 2)
        return res

    def var_blocks(self, X):
        fn = jax.vmap(lambda p,x: p.covar(x[None, :], x[None, :]), [None, 0])
        res = fn(self, X)
        diag_vec = np.diagonal(res, axis1=1, axis2=2)

        return diag_vec.T[..., None]

    @property
    def base_prior(self):
        #if self.is_base:
        if not self.hierarchical:
            return self

        return self.parent.base_prior

    @property
    def hierarchical_base_prior(self):
        if self.hierarchical:
            return self

        # when not hierarchical this should have the same behavior as base prior
        return self.base_prior



class PDE(Transform):
    def __init__(self):
        self.boundary_conditions = None
        self.boundary_by_init = False
        self.observe_data = False
        self.colocation_noise = 0.0

    def m_inf(self, x, X_s, t):
        return self.m_init

    def P_inf(self, x, X_s, t):
        return self.parent.P_inf(x, X_s, t)

    def jac(self, x, X_s, t):
        """Compute d (self.forward(x))(dx) """
        chex.assert_rank(x, 2)
        # P x D
        J =  jax.jacfwd(lambda _x: self.forward_g(_x, X_s, t))(x)[..., 0] 
        chex.assert_rank(J, 2)
        return J

    def H_jac(self, x, X_s, t):
        return self.jac(x, X_s, t)

class StackedPDE(PDE):
    def __init__(self, latent):
        self.pde_prior = objax.ModuleList(latent)
        self._parent = Independent([
            p.parent if isinstance(p, PDE)
            else p 
            for p in latent #get SDE priors
        ])
        self.num_latents =  len(latent)
        self._output_dim = sum([p.output_dim for p in latent])
    

    def m_inf(self, x, X_s, t):
        m_init_arr = []
        for i in range(self.num_latents):
            m_init_arr.append(
                self.pde_prior[i].m_inf(x, X_s, t)
            )

        return np.vstack(m_init_arr)

    def P_inf(self, x, X_s, t):
        P_inf_arr = []
        for i in range(self.num_latents):
            P_inf_arr.append(
                self.pde_prior[i].P_inf(x, X_s, t)
            )

        return to_block_diag(P_inf_arr)

    def H(self, x, X_s, t):
        H_arr = []
        for i in range(self.num_latents):
            H_arr.append(
                self.pde_prior[i].H(x, X_s, t)
            )

        return to_block_diag(H_arr)

    def forward_g(self, f, X_s, t):
        """ 
        f is of shape 3 corresponding to f, ft
        """
        # assuming dims are the same
        n_dim = f.shape[0] // self.num_latents
        g_arr = []
        for i in range(self.num_latents):
            g_arr.append(
                self.pde_prior[i].forward_g(f[i*n_dim:(i+1)*n_dim], X_s, t)
            )

        return np.hstack(g_arr)

    def psuedo_observations(self, X_s):
        z_arr = []
        for i in range(self.num_latents):
            z_arr.append(
                self.pde_prior[i].psuedo_observations(X_s)
            )

        return np.vstack(z_arr)

class TaylorLinearizedDE(PDE, LinearTransform):
    def __init__(self, latent, pde_transform, input_dim, output_dim, data_y_index=None):
        self._parent = latent
        self.pde_transform = pde_transform
        self._output_dim = output_dim
        self._input_dim = input_dim
        self.data_y_index = data_y_index
        self.latents = self.parent.latents

    def get_linear_terms(self, mu):
        b = self.pde_transform.forward(mu)[:, None]
        A = jacfwd(self.pde_transform.forward)(mu)[..., 0]

        # TODO: hacky
        if len(A.shape) == 3:
            A = A[0]
            b = b[0]


        return A, b - A@mu
        #return A, b 

    def forward(self, f):
        A, b = self.get_linear_terms(f)
        f = np.reshape(f, [-1, 1])
        return ((A @ f)+b)[:, 0]

    def transform_diagonal(self, mu, var):
        """
        Computes the pointwise of 
            F_n = W U_n
        This functions assumes that U_n are indepenent
        Hence F_n ~ N(W mu_n, W diag(var_n) W.T)
        The diagonal of this is given by:
            W mu
            W diag(sqrt(var))
        """
        chex.assert_rank([mu, var], [2, 3])

        A, b = self.get_linear_terms(mu)

        # Mixing latent functions
        mu = A @ mu + b
        var = batched_diagonal_from_XDXT(A, var[..., 0])

        # fix shapes
        var = var[..., None]

        return mu, var


    def transform(self, mu, var):
        chex.assert_rank([mu, var], [2, 3])
        chex.assert_equal([mu.shape[1], var.shape[0]], [1, 1])
        chex.assert_equal([var.shape[1], var.shape[2]], [mu.shape[0], mu.shape[0]])

        A, b = self.get_linear_terms(mu)

        g_m = A @ mu + b
        g_S = (A @ var[0] @ A.T)[None, ...]

        return g_m, g_S 

class IdentityPDE(PDE):
    def __init__(self, latent, m_init = None, d = 2, full_state = True):
        self._parent = latent
        self._output_dim = 1
        self._input_dim = self.parent.output_dim
        self.d = d # state_dim of a single latent
        self.full_state = full_state

        if self.full_state:
            self._output_dim = d*self.parent.num_latents
        else:
            self._output_dim = self.parent.num_latents

        if m_init is None:
            m_init = np.zeros(self.output_dim)[:, None]

        self.m_init = np.array(m_init)



    def m_inf(self, x, X_s, t):
        return self.m_init

    def P_inf(self, x, X_s, t):
        return self.parent.P_inf(x, X_s, t)

    def H_full_state(self, x, X_s, t):
        return np.eye(self.d*self.parent.num_latents)

    def H(self, x, X_s, t):
        if self.full_state:
            return self.H_full_state(x, X_s, t)
        else:
            raise NotImplementedError()

    def forward_g(self, f, X_s, t):
        """ 
        f is of shape 2 corresponding to f, ft
        """
        if self.full_state:
            return np.squeeze(f)
        else:
            # only extract the derivative terms
            return np.array([f[1+self.d * i] for i in range(self.parent.num_latents)])

    def psuedo_observations(self, X_s):
        # not a PDE in to enforce, so no colocation points
        # [y1, dy1, y2, dy2]
        if self.full_state:
            return np.array([onp.nan, onp.nan]*self.parent.num_latents)[:, None]
        return np.array([onp.nan]*self.parent.num_latents)[:, None]

class SimpleODE(PDE):
    def __init__(self, latent, m_init = None, full_state = False):
        self._parent = latent
        self._output_dim = 1
        self._input_dim = self.parent.output_dim

        if m_init is None:
            m_init = np.zeros(self.input_dim)[:, None]

        self.m_init = np.reshape(np.array(m_init), [np.array(m_init).shape[0], 1])
        self.full_state = full_state

    def m_inf(self, x, X_s, t):
        return self.m_init

    def P_inf(self, x, X_s, t):
        return self.parent.P_inf(x, X_s, t)

    def forward_g(self, f, X_s, t):
        """ 
        f is of shape 3 corresponding to f, ft

        df/dt = 2*t
        """
        #return f[1]-2*t
        _g =  f[1]+np.sin(t)

        if self.full_state:
            return np.hstack([f[0], _g])
        return _g

    def _H(self, x, X_s, t):
        return self.jac(x, X_s, t)

    def H_jac_full_state(self, x, X_s, t):
        H1 = self._H(x, X_s, t)
        if not self.full_state:
            # when full_state = True, forward_g is augments with an extra dimension
            #   so when calling jax.jac this will automatically compute the full state H
            # when false we compute it manually
            H0 = np.array([1.0, 0.0])[None, :]
            H1 = np.vstack([H0, H1])
        return H1

    def H_jac(self, x, X_s, t):
        return self._H(x, X_s, t)

    def H_full_state(self, x, X_s, t):
        return np.eye(2)

    def H(self, x, X_s, t):
        return np.array([1.0, 0.0])[None, :]

    def psuedo_observations(self, X_s):
        return np.array([onp.nan, 0.0])[:, None]



class Pendulum1D(PDE):
    def __init__(self, latent, g, l, train=True):
        """
        Latent must be a DifferentialOperatorJoint with a 1D differential kernel.

        Let x be the angle

        The  transform is:
            d^2 x/dt^2 + sin(x) = 0
        """
        self._parent = latent
        self._output_dim = 1
        self._input_dim = self.parent.output_dim

        self.W = np.eye(1)

        self.g_param = Parameter(
            np.array(g), 
            constraint='positive', 
            name ='Pendulum1D/g', 
            train=train
        )

        self.l_param = Parameter(
            np.array(l), 
            constraint='positive', 
            name ='Pendulum1D/l', 
            train=train
        )

    def m_inf(self, x, X_s, t):
        return np.zeros(self.input_dim)[:, None]

    def P_inf(self, x, X_s, t):
        return self.parent.P_inf(x, X_s, t)

    def forward(self, f):
        """ 
        f is of shape 3 corresponding to f, ft, ft2
        """
        t = f[0]
        dt2 = f[2]

        ls = self.g_param.value / self.l_param.value

        res = dt2 + ls * np.sin(t)

        return np.array([res])
class DampedPendulum1D(PDE):
    def __init__(self, latent, b, g, l, train=True, data_y_index=None):
        """
        Latent must be a DifferentialOperatorJoint with a 1D differential kernel.

        Let x be the angle

        The  transform is:
            d^2 x/dt^2 + sin(x) + d x/dt = 0
        """
        self._parent = latent
        self._output_dim = 1
        self.data_y_index = data_y_index

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        self.W = np.eye(1)

        self.b_param = Parameter(
            np.array(b), 
            constraint='positive', 
            name ='Pendulum1D/b', 
            train=train
        )

        self.g_param = Parameter(
            np.array(g), 
            constraint='positive', 
            name ='Pendulum1D/g', 
            train=train
        )

        self.l_param = Parameter(
            np.array(l), 
            constraint='positive', 
            name ='Pendulum1D/l', 
            train=train
        )

    def _f(self, init_x, t):
        ls = self.g_param.value / self.l_param.value
        b = self.b_param.value

        t = init_x[0]
        dt = init_x[1]

        return np.array([
            dt,
            - ls * np.sin(t) - b * dt
        ])

    def forward(self, f):
        """ 
        f is of shape 3 corresponding to f, ft, ft2
        """
        t = f[0]
        dt = f[1]
        dt2 = f[2]

        ls = self.g_param.value / self.l_param.value
        b = self.b_param.value

        res = dt2 + ls * np.sin(t) + b * dt

        return np.array([res])

class SpatialDampedPendulum(PDE):
    def __init__(self, latent, b, g, l, train=True):
        """
        Latent must be a DifferentialOperatorJoint with a 1D differential kernel.

        Let x be the angle

        The  transform is:
            d^2 x/ds^2 + sin(x) + d x/ds = 0
        """
        self._parent = latent
        self._output_dim = 1

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        self.W = np.eye(1)

        self.b_param = Parameter(
            np.array(b), 
            constraint='positive', 
            name ='Pendulum1D/b', 
            train=train
        )

        self.g_param = Parameter(
            np.array(g), 
            constraint='positive', 
            name ='Pendulum1D/g', 
            train=train
        )

        self.l_param = Parameter(
            np.array(l), 
            constraint='positive', 
            name ='Pendulum1D/l', 
            train=train
        )

    def forward(self, f):
        """ 
        f is of shape 6 corresponding to f, fs, fs2, dt, ...
        """
        t = f[0]
        ds = f[1]
        ds2 = f[2]

        ls = self.g_param.value / self.l_param.value
        b = self.b_param.value

        res = ds2 + ls * np.sin(t) + b * ds

        return np.array([res])



class HeatEquation2D(PDE, LinearTransform):
    def __init__(self, latent):
        """
        Latent must be a DifferentialOperatorJoint with a 2D differential kernel.

        The heat kernel transform is:
            df/dt - d^2f/dx^2 = 0
        """
        self._parent = latent
        self._output_dim = 1
        self._input_dim = self.parent.output_dim

        self.W = np.eye(1)

    def forward(self, f):
        """ 
        f is of shape 5 corresponding to f, ft, ft2, fx, fx2
        """
        return np.array([f[1] - f[4]])

    def _transform_mean(self, mu):

        dt = mu[1]
        dx2 = mu[4]

        return np.array([dt - dx2])

    def _transform_covar(self, var):

        N1 = int(var.shape[0]/self.input_dim)
        N2 = int(var.shape[1]/self.input_dim)

        left_idx = np.arange(N1)
        right_idx = np.arange(N2)

        Kt = var[left_idx+N1*1, :]
        Kt = Kt[:, right_idx+N2*1]

        Kx2 = var[left_idx+N1*4, :]
        Kx2 = Kx2[:, right_idx+N2*4]

        Ktx2 = var[left_idx+N1*1, :]
        Ktx2 = Ktx2[:, right_idx+N2*4]

        Kx2t = var[left_idx+N1*4, :]
        Kx2t = Kx2t[:, right_idx+N2*1]

        K =  Kt + Kx2 - Ktx2 - Kx2t

        #ensure correct rank
        chex.assert_shape(K, [N1 * self.output_dim, N2 * self.output_dim]) 

        return K

    def transform(self, mu, var):
        return self._transform_mean(mu), self._transform_covar(var)


    def mean(self, X):
        # parent is a DifferentialOperatorJoint whose outputs are:
        #   f, ft, ft2, fx, fx2 
        parent_mean = self.parent.mean_blocks(X)

        return self._transform_mean(parent_mean)

    def covar(self, X1, X2):
        parent_covar = self.parent.covar(X1, X2)

        return self._transform_covar(parent_covar)

class AllenCahn(PDE):
    def __init__(self, latent, train=True, m_init = None, m_init_dim = None, train_m_init=True, boundary_conditions=None, boundary_by_init=False, observe_data=False):

        super(AllenCahn, self).__init__()

        self._parent = latent
        self.latents = self.parent.latents # legacy reasons
        self.temporal_output_dim = 2
        self.spatial_output_dim = 2
        self.num_latents = len(self.latents)


        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        # TODO: fix
        if m_init is None:
            m_init = np.array([0.0]*m_init_dim)[:, None]
        else:
            m_init = np.array(m_init).reshape([-1, 1])

        self.m_init_param = Parameter(m_init, name=f'AllenCahn/m_init', train=train_m_init)

        self.ndt = 2
        self.nds = 2
        self.full_state = True
        self._output_dim = 4

        self.boundary_conditions = boundary_conditions
        self.boundary_by_init = boundary_by_init
        self.observe_data = observe_data

    @property
    def m_init(self):
        return self.m_init_param.value

    def H_full_state(self, x, X_s, t):
        return self.parent.H(x, X_s, t)
        #H_full_state =  np.eye(self.ndt * self.nds * X_s.shape[0])
        #return H_full_state

    def H(self, x, X_s, t):
        if self.full_state:
            return self.H_full_state(x, X_s, t)
        else:
            raise NotImplementedError()

    def H_jac(self, x, X_s, t):
        return self.jac(x, X_s, t)

    def _f(self, init_x, t):
        raise NotImplementedError()

    def forward(self, f):
        """ 
        f is of shape 3 corresponding to f, ft fx2
        """
        t = f[0]
        dt = f[1]
        dx2 = f[2]

        res = dt - 0.0001 * dx2 + 5 * (t**3) - 5 * t

        return np.array([res])

    def forward_g(self, f, X_s, t):
        """ 
        f is of shape 4 corresponding to x , dxs2, dxt, dxs2, dxt
        """
        # in [ds, space, df]
        f = np.reshape(f, [1, self.nds, X_s.shape[0], self.ndt])
        f = np.transpose(f, [0, 2, 3, 1]) # Nt, Ns, ndt, nds
        f = np.reshape(f, [1, X_s.shape[0], self.nds*self.ndt]) #N x d
        res =  jax.vmap(lambda _f: self.forward([_f[0], _f[2], _f[1]]))(f[0])
        return res

    def jac(self, x, X_s, t):
        """Compute d (self.forward(x))(dx) """
        chex.assert_rank(x, 2)
        # P x D
        J =  jax.jacfwd(lambda _x: self.forward_g(_x, X_s, t))(x)[:, 0, :, 0]
        chex.assert_rank(J, 2)
        return J

    def psuedo_observations(self, X_s):
        Ns = X_s.shape[0]
        return np.array([0.0]*Ns)[:, None]


class _LorenzSystemX(PDE):
    def __init__(self, latent, sigma, train=True):

        self._parent = latent
        self._output_dim = 1

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim


        self.sigma_param = Parameter(
            np.array(sigma), 
            name ='LorenzSystem/sigma', 
            train=train
        )

    def forward(self, f):
        """ 
        f is of shape 6 corresponding to x, xt, y, yt, z, zt 
        """
        x, xt, y, yt, z, zt = f
        sigma = self.sigma_param.value

        return (xt - sigma * (y - x))[:, None]

class _LorenzSystemY(PDE):
    def __init__(self, latent, rho, train=True):
        self._parent = latent
        self._output_dim = 1

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        self.rho_param = Parameter(
            np.array(rho), 
            name ='LorenzSystem/rho', 
            train=train
        )

    def forward(self, f):
        """ 
        f is of shape 6 corresponding to x, xt, y, yt, z, zt 
        """
        x, xt, y, yt, z, zt = f
        rho = self.rho_param.value

        return (yt - x * (rho - z) + y)[:, None]

class _LorenzSystemZ(PDE):

    def __init__(self, latent, beta, train=True):
        self._parent = latent
        self._output_dim = 1

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        self.beta_param = Parameter(
            np.array(beta), 
            name ='LorenzSystem/beta', 
            train=train
        )

    def forward(self, f):
        """ 
        f is of shape 6 corresponding to x, xt, y, yt, z, zt 
        """
        x, xt, y, yt, z, zt = f
        beta = self.beta_param.value

        return (zt - x * y + beta * z)[:, None]


def LorenzSystem(latent, sigma, rho, beta, train=True):
    """
    THe lorenz stystem describes the following system of equations

        dx/dt = sig * (y-x)
        dy/dt = x * ( rho - z) - y
        dz/dt = x * y - beta * z
    """
    return [
        _LorenzSystemX(latent[0], sigma, train=train), 
        _LorenzSystemY(latent[1], rho, train=train), 
        _LorenzSystemZ(latent[2], beta, train=train), 
    ]


class _LotkaVolterraSystemX(PDE):
    """ Prey Component"""

    def __init__(self, latent, alpha, beta, train=True, data_y_index=[0]):
        self._parent = latent
        self._output_dim = 1
        self.data_y_index = data_y_index

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        self.alpha_param = Parameter(
            np.array(alpha), 
            name ='LotkaVolterraSystem/alpha', 
            constraint='positive', 
            train=train
        )

        self.beta_param = Parameter(
            np.array(beta), 
            name ='LotkaVolterraSystem/beta', 
            constraint='positive', 
            train=train
        )

    def _dfdt(self, f, X_s=None, t=None):
        """ Evalutate df/dt """
        x, xt, y, yt = f
        alpha = self.alpha_param.value
        beta = self.beta_param.value

        return alpha * x - beta * x * y

    def forward(self, f):
        """ 
        f is of shape 4 corresponding to x, xt, y, yt
        """
        x, xt, y, yt = f
        return (xt - self._dfdt(f))[:, None]


    def forward_g(self, f, X_s, t):
        """ 
        f is of shape 4 corresponding to x, xt, y, yt
        """
        return self.forward(f)

class _LotkaVolterraSystemY(PDE):
    """ Predator Component"""

    def __init__(self, latent, delta, gamma, train=True, data_y_index=[1]):
        self._parent = latent
        self._output_dim = 1
        self.data_y_index = data_y_index

        if self.parent is None:
            self._input_dim = None
        else:
            self._input_dim = self.parent.output_dim

        self.delta_param = Parameter(
            np.array(delta), 
            name ='LotkaVolterraSystem/delta', 
            constraint='positive', 
            train=train
        )

        self.gamma_param = Parameter(
            np.array(gamma), 
            name ='LotkaVolterraSystem/gamma', 
            constraint='positive', 
            train=train
        )

    def _dfdt(self, f, X_s=None, t=None):
        """ Evalutate df/dt """
        x, xt, y, yt = f
        delta = self.delta_param.value
        gamma = self.gamma_param.value

        return delta * x * y  - gamma * y

    def forward(self, f):
        """ 
        f is of shape 4 corresponding to x, xt, y, yt
        """
        x, xt, y, yt = f

        return (yt - self._dfdt(f))[:, None]

    def forward_g(self, f, X_s, t):
        """ 
        f is of shape 4 corresponding to x, xt, y, yt
        """
        return self.forward(f)

class LotkaVolterra(PDE):
    def __init__(self, latent,  alpha, beta, delta, gamma, m_init=None, train=True, full_state = False, train_m_init=False, boundary_conditions=None, boundary_by_init=False, observe_data=False):
        super(LotkaVolterra, self).__init__()

        self.components = objax.ModuleList([
            _LotkaVolterraSystemX(None, alpha, beta, train=train), 
            _LotkaVolterraSystemY(None, delta, gamma, train=train), 
        ])
        if m_init is None:
            m_init = np.array([0.0, 0.0, 0.0, 0.0])[:, None]
        else:
            m_init = np.array(m_init).reshape([4, 1])

        self._parent = latent

        self.full_state = full_state

        if self.full_state:
            self._output_dim = 4
        else:
            self._output_dim = 2

        self.m_init_param = Parameter(m_init, name=f'LotkaVolterra/m_init', train=train_m_init)
        self.boundary_conditions = boundary_conditions
        self.boundary_by_init = boundary_by_init
        self.observe_data = observe_data

    @property
    def m_init(self):
        return self.m_init_param.value

    def m_inf(self, x, X_s, t):
        if self.boundary_by_init:
            return self.m_init
        return np.zeros_like(self.m_init)

    def P_inf(self, x, X_s, t):
        if self.boundary_by_init:
            return self.parent.P_inf(x, X_s, t)

        return np.eye(self.output_dim)*1e6

    def _dfdt(self, f, X_s=None, t=None):
        """ Evalutate df/dt """
        f0 = np.squeeze(self.components[0]._dfdt(f))
        f1 = np.squeeze(self.components[1]._dfdt(f))
        return np.array([f0, f1])
        

    def forward(self, f):
        """ 
        args:
            f: (D x 1) : 
        """
        chex.assert_rank(f, 2)

        y0 = np.squeeze(self.components[0].forward(f))
        y1 = np.squeeze(self.components[1].forward(f))

        if self.full_state:
            return np.array([f[0][0], y0, f[2][0], y1])

        return np.array([y0, y1])

    def forward_g(self, f, X_s, t):
        """ 
        f is of shape 4 corresponding to x, xt, y, yt
        """
        return self.forward(f)

    def _H(self, x, X_s, t):
        return self.jac(x, X_s, t)

    def H_jac_full_state(self, x, X_s, t):
        H01 = self._H(x, X_s, t)

        H01 = np.array([1.0, 0.0, 0.0, 0.0])[None, :]
        H03 = np.array([0.0, 0.0, 1.0, 0.0])[None, :]
        H = np.vstack([
            H01, 
            H01[[0], :],
            H03, H01[[1], :]
        ])
        return H

    def H_jac(self, x, X_s, t):
        return self._H(x, X_s, t)

    def H_full_state(self, x, X_s, t):
        return np.eye(4)

    def H(self, x, X_s, t):
        if self.full_state:
            return self.H_full_state(x, X_s, t)
        else:
            return np.eye(4)[[0, 2], :]

    def psuedo_observations(self, X_s):
        # [y1, dy1, y2, dy2]
        if self.full_state:
            return np.array([onp.nan, 0.0, onp.nan, 0.0])[:, None]
        return np.array([0.0, 0.0])[:, None]

    

def LotkaVolterraSystem(latent, alpha, beta, delta, gamma, train=True, data_y_index=[0, 1]):
    """
    THe lorenz stystem describes the following system of equations

        dx/dt = alpha *x - beta *x * y
        dy/dt = delta * x * y  - gamma * y
    """
    return [
        _LotkaVolterraSystemX(latent[0], alpha, beta, train=train, data_y_index=[data_y_index[0]]), 
        _LotkaVolterraSystemY(latent[1], delta, gamma, train=train, data_y_index=[data_y_index[1]]), 
    ]


