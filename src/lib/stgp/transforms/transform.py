"""
Base transform class.

Implements
    Transform: the base class of all transformations
    NonLinearTransform
    LinearTransform
    Joint
    Independent
"""
from ..core import Prior, GPPrior, Model
from ..utils.utils import ensure_module_list, can_batch, get_batch_type, _ensure_str
from batchjax import batch_or_loop, BatchType
from ..computation.matrix_ops import to_block_diag, batched_diagonal_from_XDXT, get_block_diagonal
from ..core import Block

import jax
import jax.numpy as np
import objax
import chex

from typing import List, Optional

import warnings

class Transform(GPPrior):
    """
    All transforms must define:
        output_dim: number of latent gp outputs
        input_dim: number of input gps
        parent: pointer to parent gp / list of gps
        forward: the transformation
        base_prior: return the base prior

    Linear Transforms must implement
        transform(mu, var): transform an input gaussian
        in_block_dim: a property defining the size of the required blocks of covariance 
        out_block_dim: a property defining the size of the transformed covariance block size

    When a transform defines a base prior it must additionally define
        get_sparsity_list
        get_Z
        get_sparsity
    
    """
    def __init__(self):
        self._output_dim = None
        self._input_dim = None

        # parent obj that is being transformed
        self._parent = None
        self.data_y_index = None

    @property
    def full_transform(self):
        return False


    def transform_diagonal(self, mu, var):
        """ Transform a diagonal gaussian dist. This should matain rank 2."""
        raise NotImplementedError(f'{self}')

    def transform(self, mu, var):
        """ 
        Transform a full gaussian dist 

        mu and var are rank 2, should return rank 2.
        """
        
        raise NotImplementedError(f'{self}')

    def forward(self, x):
        """Compute f=T(x)."""
        raise NotImplementedError()

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        raise NotImplementedError()

    @property
    def is_base(self):
        return False

    @property
    def hierarchical_base_prior(self):
        return self.parent.hierarchical_base_prior

    @property
    def base_prior(self):
        """
        A transform is paced on top of a GP prior. This returns that base GP prior.
        """
        return self.parent.base_prior

    @property
    def num_outputs(self): raise RuntimeWarning('num_outputs has been removed. Use output_dim instead.')

    @property
    def output_dim(self): return self._output_dim

    @property
    def input_dim(self): return self._input_dim

    @property
    def parent(self): return self._parent

    def mean(self, XS): raise NotImplementedError()
    def mean_blocks(self, XS):raise NotImplementedError()
    def var(self, XS):raise NotImplementedError()
    def var_blocks(self, XS):raise NotImplementedError()
    def covar_blocks(self, X1, X2):raise NotImplementedError()
    def covar(self, X1, X2):raise NotImplementedError()
    def full_var(self, X):raise NotImplementedError()

class NonLinearTransform(Transform):
    def __init__(self, latent):
        self._parent = latent

class LinearTransform(Transform):
    def __init__(self, latent):
        self._parent = latent

    @property
    def in_block_dim(self) -> Block:
        return None

    @property
    def out_block_dim(self) -> Block:
        return None

    @property
    def base_prior(self):
        return self.parent.base_prior

    def get_sparsity_list(self):
        return self.parent.get_sparsity_list()

class Joint(Transform):
    def __init__(self, parent):
        super(Joint, self).__init__()
        self._parent = parent

    @property
    def is_base(self):
        return True

class Independent(Transform):
    def __init__(
        self, 
        latents: Optional[List['Model']] = None, 
        latent: Optional['Model'] = None,
        prior=True
    ) -> None:
        """
        Args:
            prior: bool -- Indicates whether latents are priors or posteriors
        """ 
        super().__init__()

        self.prior = prior

        if (latents is None) and (latent is None):
            raise RuntimeError('Latents must be passed')

        if (latent is not None) and (latents is not None):
            raise RuntimeError('Only latent or latents must be passed')

        if latent:
            # Standardize input to ease implementation
            latents = [latent]

        if prior:
            self._output_dim = sum([p.output_dim for p in latents])
        else:
            self._output_dim = latent.output_dim

        self._input_dim = self.output_dim
        self._parent = ensure_module_list(latents)

        self.whiten_space = False


    def transform_diagonal(self, mu, var): return mu, var
    def transform(self, mu, var): return mu, var

    @property
    def latents(self):
        return self.parent

    @property
    def is_base(self):
        return True

    @property
    def base_prior(self):
        return self

    @property
    def hierarchical_base_prior(self):
        return self

    def get_sparsity_list(self):
        """ Collect all sparsity objects of all latents in a flat list """
        sparsity_list = []

        for p in self.parent:
            sparsity_list += p.get_sparsity_list()

        # hack for now
        if type(sparsity_list[0]) is list:
            # assuming only one latent per mean field
            sparsity_list = [s[0] for s in sparsity_list]

        return sparsity_list

    def get_sparsity(self): return self.get_sparsity_list()


    def get_Z_stacked(self):
        Z_arr = batch_or_loop(
            lambda latent:  latent.get_Z_blocks(),
            [self.parent],
            [0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        # each latent.get_b_Z() is of rank 3
        #chex.assert_rank(Z_arr, 4)

        return Z_arr

    def get_Z_blocks(self):
        Z_arr = self.get_Z_stacked()

        # each latent.get_b_Z() is of rank 3
        chex.assert_rank(Z_arr, 4)

        # convert to rank 3 by stacking
        Z_arr = np.vstack(Z_arr)

        chex.assert_rank(Z_arr, 3)
        return Z_arr

    def get_Z(self):
        Z_arr = self.get_Z_blocks()
        Z =  np.vstack(Z_arr)
        chex.assert_rank(Z, 2)

        return Z

    def mean_blocks(self, X1: np.ndarray) -> np.ndarray:
        mean = batch_or_loop(
            lambda X1, latent:  latent.mean(X1),
            [X1, self.parent],
            [None, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        mean = np.array(mean)

        mean = np.reshape(
            mean, 
            [self.output_dim, X1.shape[0], 1]
        )

        return mean

    def mean(self, X1):
        return np.vstack(self.mean_blocks(X1))

    def covar_blocks(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        k_arr = batch_or_loop(
            lambda X1, X2, latent:  latent.covar(X1, X2),
            [X1, X2, self.parent],
            [None, None, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        k_arr = np.array(k_arr)

        if False:
            k_arr = np.reshape(
                k_arr, 
                [self.output_dim, X1.shape[0], X2.shape[0]]
            )

        return k_arr

    def covar(self, X1, X2):
        c_blocks = self.covar_blocks(X1, X2)
        return to_block_diag(c_blocks)

    def b_covar_blocks(self, X1, X2):
        # Independent can be over a list of latent GPs or priors
        #  hence we call latent.b_covar, which in the case of a single gp is the same
        #  sa latent.covar
        c_blocks = batch_or_loop(
            lambda x1, x2, latent:  latent.b_covar(x1, x2),
            [X1, X2, self.parent],
            [0, 0, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        return c_blocks

    def b_covar(self, X1, X2):

        c_blocks = self.b_covar_blocks(X1, X2)

        res = to_block_diag(c_blocks)

        return res

    def b_mean(self, X1):
        mean_blocks = batch_or_loop(
            lambda x1, latent:  latent.b_mean(x1),
            [X1, self.parent],
            [0, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )
        return np.vstack(mean_blocks)

    def b_mean_blocks(self, X1):
        mean_blocks = batch_or_loop(
            lambda x1, latent:  latent.b_mean_blocks(x1),
            [X1, self.parent],
            [0, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )
        return np.vstack(mean_blocks)

    def var_blocks(self, X1: np.ndarray) -> np.ndarray:
        var = batch_or_loop(
            lambda X1, latent:  latent.var(X1),
            [X1, self.parent],
            [None, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        var = np.array(var)
        var = np.reshape(
            var, 
            [self.output_dim, X1.shape[0], 1]
        )

        return var

    def var(self, X):
        v_blocks = self.var_blocks(X)
        v_stacked =  np.vstack(v_blocks)
        chex.assert_shape(v_stacked, [X.shape[0]*self.output_dim, 1])
        return v_stacked

    def full_var_blocks(self, X1: np.ndarray) -> np.ndarray:
        return self.covar_blocks(X1, X1)

    @property
    def temporal_output_dim(self):
        """
        Only return `f'
        """
        return self.output_dim

    @property
    def spatial_output_dim(self):
        # TODO: hack for deadline
        try:
            # use loop so that we get a list of int outputs
            if hasattr(self.parent[0], 'kernel'):
                fn = lambda  latent:  latent.kernel.spatial_output_dim
            else:
                fn = lambda  latent:  latent.spatial_output_dim
            return [fn(latent) for latent in self.parent]
        except Exception as e:
            return [1]

    def state_space_dim(self):
        # use loop so that we get a list of int outputs
        if hasattr(self.parent[0], 'kernel'):
            fn = lambda  latent:  latent.kernel.state_space_dim()
        else:
            fn = lambda  latent:  latent.state_space_dim()

        return [fn(latent) for latent in self.parent]

    def state_space_representation(self, X_s):
        F_blocks, L_blocks, Qc_blocks, H_blocks, m_inf_blocks, P_inf_blocks = batch_or_loop(
            lambda x_s, latent:  latent.kernel.to_ss(x_s),
            [X_s, self.parent],
            [None, 0],
            dim = self.output_dim,
            out_dim = 6,
            batch_type = get_batch_type(self.parent)
        )

        F = to_block_diag(F_blocks)
        L = to_block_diag(L_blocks)
        Qc = to_block_diag(Qc_blocks)
        m_inf = np.vstack(m_inf_blocks)
        P_inf = to_block_diag(P_inf_blocks)
        H = to_block_diag(H_blocks)

        return F, L, Qc, H, m_inf, P_inf

    def expm(self, dt, X_s):
        # TODO: clean up at some point (see self.P_inf)
        if hasattr(self.parent[0], 'kernel'):
            fn = lambda d, x_s, latent:  latent.kernel.expm(dt, x_s)
        else:
            fn = lambda d, x_s, latent:  latent.expm(dt, x_s)

        A_blocks = batch_or_loop(
            fn,
            [dt, X_s, self.parent],
            [None, None, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        return to_block_diag(A_blocks)

    def P_inf(self, x, X_s, t):

        # TODO: clean up at some point
        # this is just a way to support wrapping both SDE_GPs and Transforms of them in an Independent
        #  these should have the same api and then this would not be necessary

        if hasattr(self.parent[0], 'kernel'):
            fn = lambda x, X_s, t, latent:  latent.kernel.P_inf(x, X_s, t)
        else:
            fn = lambda x, X_s, t, latent:  latent.P_inf(x, X_s, t)

        P_inf_blocks = batch_or_loop(
            fn,
            [x, X_s, t, self.parent],
            [None, None, None, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        return to_block_diag(P_inf_blocks)

    def m_inf(self, x, X_s, t):

        # TODO: clean up at some point (see self.P_inf)

        if hasattr(self.parent[0], 'kernel'):
            fn = lambda x, X_s, t, latent:  latent.kernel.m_inf(x, X_s, t)
        else:
            fn = lambda x, X_s, t, latent:  latent.m_inf(x, X_s, t)

        m_inf_blocks = batch_or_loop(
            fn,
            [x, X_s, t, self.parent],
            [None, None, None, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        return np.hstack(m_inf_blocks)

    def H(self, x, X_s, t):

        # TODO: clean up at some point (see self.P_inf)

        if hasattr(self.parent[0], 'kernel'):
            fn = lambda x, X_s, t, latent:  latent.kernel.H(x, X_s, t)
        else:
            fn = lambda x, X_s, t, latent:  latent.H(x, X_s, t)

        H_inf_blocks = batch_or_loop(
            fn,
            [x, X_s, t, self.parent],
            [None, None, None, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        return np.hstack(H_inf_blocks)

    def Q(self, dt_k, A_k, P_inf, X_spatial=None):
        #warnings.warn('HACK IN INDEPENDENT TRANSFORM Q')
        # A_k, P_inf SHOULD be in block form here, but they are not...
        # for now it is okay as IWP handles it self (?) and all other kernels 
        # will be the same across latent functions so this will return the same thing
        #return self.parent[0].kernel.Q(dt_k, A_k, P_inf, X_spatial=X_spatial)

        if X_spatial is None:
            Ns = 1
        else:
            Ns = X_spatial.shape[0]

        if hasattr(self.parent[0], 'kernel'):
            fn = lambda dt, A, P, Xs, latent:  latent.kernel.Q(dt, A, P, X_spatial=Xs)
        else:
            fn = lambda dt, A, P, Xs, latent:  latent.Q(dt, A, P, X_spatial=Xs)

        # all of this is just a way to bypass the fact that A_k and P_inf are not blocks
        # i am trying to figure what the blocks SHOULD have been
        # extracting them, and then procedding as normal

        dt_dims = self.state_space_dim()
        ds_dims = self.spatial_output_dim

        if type(dt_dims) is list:
            if type(dt_dims[0]) is list:
                block_dim = sum(dt_dims[0])*sum(ds_dims[0])
            else:
                block_dim = dt_dims[0]*ds_dims[0]
        else:
            block_dim = dt_dims*ds_dims

        block_dim = block_dim*Ns

        A_k = get_block_diagonal(A_k, block_dim)
        P_inf = get_block_diagonal(P_inf, block_dim)

        Q_blocks = batch_or_loop(
            fn,
            [dt_k, A_k, P_inf, X_spatial, self.parent],
            [None, 0, 0, None, 0],
            dim = self.output_dim,
            out_dim = 1,
            batch_type = get_batch_type(self.parent)
        )

        return to_block_diag(Q_blocks)

    def fix(self):
        for q in self.parent:
            q.fix()

    def release(self):
        for q in self.parent:
            q.release()



class MultiOutput(Transform):
    def __init__(self, parent):
        """
        Construct a multi-output prior by stacking f horizontally

        However all objects in parent must share the SAME base prior
        """
        # assert that all base_priors in parent are the same
        self.check_base_priors_are_same(parent)

        self._parent = objax.ModuleList(
            parent
        )

        self._output_dim = sum([p.output_dim for p in self.parent])

    def check_base_priors_are_same(self, parent):
        base_prior_list = [p.base_prior for p in parent]
        all_equal = np.all(np.array([base_prior_list[i] == base_prior_list[i-1] for i in range(1, len(base_prior_list))]))

        if not all_equal:
            raise RuntimeError('All elements passed to MultiOutput must share the same base prior')

    def transform(self, mu, var):
        mu_arr = []
        var_arr = []

        # each must return a single output
        for p in self.parent:
            _m, _v = p.transform(mu, var)
            mu_arr.append(_m)
            var_arr.append(_v)

        # TODO: this assuming a single output but if we return the list here we can generalise this
        return np.vstack(mu_arr), to_block_diag(var_arr)

    def forward(self, f):
        res = []
        for p in self.parent:
            res.append(
                p.forward(f)
            )
        return np.hstack(res)

    @property
    def base_prior(self):
        # all objects in parent share the same base_prior so we can just return the first one
        return self.parent[0].base_prior


    @property
    def hierarchical_base_prior(self):
        return self.parent[0].hierarchical_base_prior


class NonLinearIdentity(NonLinearTransform):
    def __init__(self, parent):
        super(NonLinearIdentity, self).__init__(parent)
        self._output_dim = 1

    """ For debugging purposes """
    def forward(self, x):
        return x

    def inverse(self, f):
        return f

class One2One(NonLinearTransform):
    def __init__(self, parent: 'Transform', transform_arr: list):
        self._parent = parent
        self.transform_arr = objax.ModuleList(transform_arr)
        self._output_dim = len(self.transform_arr)
        self._input_dim = self.output_dim

    @property
    def base_prior(self):
        return self.parent.base_prior

    def forward(self, f):
        num_outputs = len(self.transform_arr)

        f_prop = np.reshape(f, [num_outputs, 1])

        f_transformed = batch_or_loop(
            lambda t_fn, f_p: t_fn.forward(f_p),
            [ self.transform_arr, f_prop ],
            [ 0, 0],
            dim = num_outputs,
            out_dim = 1,
            batch_type = get_batch_type(self.transform_arr)
        )
        f_transformed = np.array(f_transformed)


        f_transformed = np.reshape(f_transformed, [num_outputs, 1])

        return f_transformed

class LatentSpecific(Transform):
    """ A transform that can only be applied to a single latent function """
    pass

class ElementWiseTransform(LatentSpecific):
    def __init__(self):
        super(ElementWiseTransform, self).__init__()
        self._input_dim = 1
        self._num_outputs = 1

