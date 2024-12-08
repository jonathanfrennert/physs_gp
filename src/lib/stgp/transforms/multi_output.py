"""Multi-output/task specific transforms."""
from .transform import Transform, LinearTransform, NonLinearTransform, Independent
import typing
from typing import List, Optional, Union
import jax
import jax.numpy as np
import numpy as onp
import objax
import chex
from ..computation.parameter_transforms import get_correlation_cholesky, correlation_transform, inv_correlation_transform,softplus, probit
from ..computation.matrix_ops import batched_diagonal_from_XDXT, lower_triangle
from .. import Parameter

class GPRN_Base(NonLinearTransform):
    def __init__(self, W, f, input_dim: int = None, output_dim: int = None):

        super(NonLinearTransform, self).__init__()

        # Flatten W into a vector - row major ordering
        W_vec = [w for W_p in W for w in W_p]

        # Flatten latents to fit into VI framework
        self._parent = Independent(
            latents = f+W_vec,
            prior = True
        )

        self._input_dim = len(f)
        self._output_dim = len(W)

    @property
    def base_prior(self):
        return self.parent.base_prior

    @property
    def forward(self, f):
        raise NotImplementedError()

class GPRN(GPRN_Base):
    def forward(self, f):
        # f has the same ordering as self.latents
        latent_f = f[:self.input_dim]
        latent_W = f[self.input_dim:]

        latent_f = np.reshape(latent_f, [self.input_dim, 1])

        # W is in row-major ordering
        latent_W = latent_W.reshape(
            self.output_dim,
            self.input_dim,
            order='C'
        )

        return (latent_W @ latent_f)

class GPRN_Exp(GPRN_Base):
    def forward(self, f):
        # f has the same ordering as self.latents
        latent_f = f[:self.input_dim]
        latent_W = f[self.input_dim:]

        # W is in row-major ordering
        latent_W = latent_W.reshape(
            self.output_dim,
            self.input_dim,
            order='C'
        )

        latent_f = np.reshape(latent_f, [self.input_dim, 1])

        # Element wise exponential to force W to be positive
        #return (np.exp(latent_W) @ latent_f)[:, 0]
        return (softplus(latent_W) @ latent_f)[:, 0]

class GPRN_LDL(GPRN_Base):

    def __init__(self, W_vec, f, input_dim: int = None, output_dim: int = None):

        #super(GPRN_LDL, self).__init__()

        # Flatten latents to fit into VI framework
        self._parent = Independent(
            latents = f+W_vec,
            prior = True
        )

        self._input_dim = len(f)
        self._output_dim = self.input_dim

    def forward(self, f):
        # f has the same ordering as self.latents
        latent_f = f[:self.input_dim]
        latent_W = f[self.input_dim:]


        P = self.output_dim
        Q = self.input_dim
        tri = np.eye(P, Q)

        num_latents = f.shape[0]

        latent_W = np.reshape(latent_W, [num_latents - self.input_dim])
        latent_f = np.reshape(latent_f, [self.input_dim, 1])

        mixing_matrix = tri.at[np.tril_indices(P, -1, Q)].set(latent_W)

        return (mixing_matrix @ latent_f)

class GPRN_DRD(GPRN_Base):

    def __init__(self, W_vec, f, a=None, variances =  None, input_dim: int = None, output_dim: int = None):

        #super(GPRN_DRD, self).__init__()

        self._input_dim = len(f)
        self._output_dim = self.input_dim

        # When using LMC_corr the mixing matrix must be square
        self.P = self.output_dim
        self.Q = int(self.P*(self.P-1)/2)

        if a is None:
            a = 1.0

        self.a = a

        # Set defaults
        if variances is None:
            variances = np.ones(self.P)
        else:
            variances = np.array(variances)

        # Setup Parameters
        self.variances = Parameter(variances, constraint='positive', name='GPRN_DRD/variance', train=True)

        # Flatten latents to fit into VI framework
        self._parent = Independent(
            latents = f+W_vec,
            prior = True
        )

    def forward(self, f):
        # f has the same ordering as self.latents

        num_latents = f.shape[0]

        latent_f = f[:self.input_dim]
        latent_W = f[self.input_dim:]

        latent_f = np.reshape(latent_f, [self.input_dim, 1])
        latent_W = np.reshape(latent_W, [num_latents - self.input_dim])

        correlation_cholesky =  get_correlation_cholesky(
            correlation_transform(latent_W, self.a), 
            self.P, 
            self.Q
        )

        var_diag = np.diag(self.variances.value)

        return (var_diag @ correlation_cholesky @ latent_f)

class GPRN_DRD_EXP(GPRN_Base):
    def __init__(self, v, W_vec, f, a=None, variances =  None, input_dim: int = None, output_dim: int = None):

        #super(GPRN_DRD, self).__init__()

        self._input_dim = len(f)
        self._output_dim = self.input_dim

        # When using LMC_corr the mixing matrix must be square
        self.P = self.output_dim
        self.Q = int(self.P*(self.P-1)/2)

        if a is None:
            self.a = 1.0

        # Flatten latents to fit into VI framework
        self._parent = Independent(
            latents = v+f+W_vec,
            prior = True
        )

    def forward(self, f):
        # f has the same ordering as self.latents
        latent_v = f[:self.output_dim]
        latent_f = f[self.output_dim:self.output_dim+self.input_dim]
        latent_W = f[self.output_dim+self.input_dim:]


        latent_v = np.reshape(latent_v, [self.input_dim])
        latent_f = np.reshape(latent_f, [self.input_dim, 1])
        latent_W = np.reshape(latent_W, [-1]) # ensure rank 1

        correlation_cholesky =  get_correlation_cholesky(
            correlation_transform(latent_W, self.a), 
            self.P, 
            self.Q
        )

        var_diag = np.diag(softplus(latent_v))

        return (var_diag @ correlation_cholesky @ latent_f)


class LMC_Base(LinearTransform):
    """
    Inherits
        num_outputs
        num_latents
    """
    def __init__(self, latents, input_dim: int, output_dim: int):
        # Allow passing a list of prior models and transformed model
        if type(latents) is list:
            self._parent = Independent(latents=latents, prior=True)
        else:
            self._parent = latents

        self._input_dim = input_dim
        self._output_dim = output_dim

    @property
    def base_prior(self):
        return self.parent.base_prior

    @property
    def W(self):
        raise NotImplementedError()

    def mean(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        P = self.output_dim

        mean = np.zeros([P, N1])

        mean = np.hstack(mean)[:, None]
        chex.assert_shape(mean, [P*N1, 1])

        return mean

    def covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        N2 = X2.shape[0]

        mixing_matrix = self.W

        Q = self.input_dim
        P = self.output_dim

        W1 = np.kron(mixing_matrix, np.eye(N1))
        W2 = np.kron(mixing_matrix, np.eye(N2))

        K_bdiag = self.parent.covar(X1, X2)
        chex.assert_shape(K_bdiag, [Q*N1, Q*N2])

        covar = W1 @ K_bdiag @ W2.T
        chex.assert_shape(covar, [P*N1, P*N2])

        return covar

    def full_var(self, X):
        return self.covar(X, X)

    def var(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        Q = self.input_dim
        P = self.output_dim

        mixing_matrix = self.W

        K_diag = self.parent.var_blocks(X1)[..., 0]
        chex.assert_shape(K_diag, [Q, N1])

        W = mixing_matrix**2

        #P x N
        covar = W @ K_diag
        chex.assert_shape(covar, [P, N1])

        # PN
        covar = np.hstack(covar)[:, None]
        chex.assert_shape(covar, [P*N1, 1])

        return covar

    def forward(self, f):
        f = np.reshape(f, [-1, 1])
        return (self.W @ f)[:, 0]

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

        W = self.W

        # Mixing latent functions
        mu = W @ mu 
        var = batched_diagonal_from_XDXT(W, var[..., 0])

        # fix shapes
        var = var[..., None]

        return mu, var


    def transform(self, mu, var):
        chex.assert_rank([mu, var], [2, 3])
        chex.assert_equal([mu.shape[1], var.shape[0]], [1, 1])
        chex.assert_equal([var.shape[1], var.shape[2]], [mu.shape[0], mu.shape[0]])

        W = self.W

        return W @ mu, (W @ var[0] @ W.T)[None, ...]


# TODO: implement ICM

class LMC(LMC_Base):
    def __init__(
        self, 
        latents: Optional[Union[List['Model'], Transform]]=None, 
        output_dim: Optional[int]=None, 
        input_dim: Optional[int]=None, 
        W: Optional[np.ndarray] = None
    ):
        if type(latents) == list:
            latents = Independent(latents)

        if input_dim is None:
            input_dim = latents.input_dim

        super().__init__(latents, input_dim=input_dim, output_dim=output_dim)

        if W is None:
            W = np.eye(self.output_dim, self.input_dim)

        # Setup correlation matrix variables
        self._W = Parameter(np.array(W), name='W')


    @property
    def W(self):
        return self._W.value

class LMC_LDL(LMC_Base):
    def __init__(
        self, 
        latents: Optional[Union[List['Model'], Transform]]=None, 
        output_dim: Optional[int]=None, 
        input_dim: Optional[int]=None, 
        W: Optional[np.ndarray] = None
    ):
        if type(latents) == list:
            latents = Independent(latents)

        super().__init__(latents, input_dim=latents.output_dim, output_dim=output_dim)

        self._num_latents = self.input_dim

        # Setup correlation matrix variables
        num_vars = int(self.output_dim*(self.output_dim-1)/2)
        self.z_arr = Parameter(np.zeros(num_vars), name='LMC_Unit_Tri/Z_arr', train=True)

    @property
    def W(self):
        P = self.output_dim
        Q = self.input_dim

        tri = np.eye(P, Q)
        mixing_matrix = tri.at[np.tril_indices(P, -1, Q)].set(self.z_arr.value)

        return mixing_matrix


class LMC_DRD(LMC_Base):
    def __init__(
        self, 
        latents: Optional[Union[List['Model'], Transform]]=None, 
        output_dim: Optional[int]=None, 
        variances: Optional[np.ndarray] = None,
        mixing_weights: Optional[np.ndarray] = None,
        a: Optional[float] = None
    ):

        if type(latents) == list:
            latents = Independent(latents)

        super().__init__(latents, input_dim=latents.output_dim, output_dim=output_dim)

        # When using LMC_corr the mixing matrix must be square
        self.P = self.output_dim
        self.Q = int(self.P*(self.P-1)/2)

        # Set defaults
        if variances is None:
            variances = np.ones(self.P)

        if mixing_weights is None:
            mixing_weights = np.zeros(self.Q)

        if a is None:
            self.a = 1.0

        # Setup Parameters
        self.variances = Parameter(variances, constraint='positive', name='LMC_Corr/variance')

        self.z_arr = Parameter(
            mixing_weights,
            constraint_fn=lambda x: correlation_transform(x, self.a), 
            inv_constraint_fn=lambda x: inv_correlation_transform(x, self.a), 
            name='LMC_Corr/z_arr'
        )
        

    @property
    def W(self):
        z_arr = self.z_arr.value
        correlation_cholesky =  get_correlation_cholesky(z_arr, self.P, self.Q)

        var_diag = np.diag(self.variances.value)

        return var_diag @ correlation_cholesky
