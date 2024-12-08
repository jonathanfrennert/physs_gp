class _LinearTransform(Transform):
    """
    All linear transforms support .W returns the mixing matrix
    """

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
        W = self.W

        # Mixing latent functions
        mu = W @ mu[..., 0] 
        var = batched_diagonal_from_XDXT(W, var[..., 0])


        # fix shapes
        mu = mu[..., None]
        var = var[..., None]

        return mu, var

    def mean(self, X1):
        """ Output shape [P, N1]. """
        raise NotImplementedError()

    def vec_mean(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        P = self.num_outputs

        mean = self.mean(X1)
        mean = np.vstack(mean)

        chex.assert_shape(mean, [N1*P, 1])
        return mean

    def covar(self, X1, X2):
        """ Output shape [P, N1, N2]. """
        raise NotImplementedError()

    def full_covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        k_arr = self.covar(X1, X2)
        k =  jax.scipy.linalg.block_diag(*k_arr)

        chex.assert_shape(
            k,
            [self.num_latents*X1.shape[0], self.num_latents*X2.shape[0]]
        )

        return k

    def var(self, X1):
        """ Output shape [P, N1]. """
        raise NotImplementedError()

    def vec_var(self, X1: np.ndarray) -> np.ndarray:
        N1 = X1.shape[0]
        P = self.num_outputs
        var = self.var(X1)

        var = np.hstack(var)[:, None]

        chex.assert_shape(var, [N1*P, 1])
        return var

    def full_var(self, X1):
        """ Output shape [PxN1, PxN1]. """
        raise NotImplementedError()




class Independent(LinearTransform):
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
        self.is_base = True

        if (latents is None) and (latent is None):
            raise RuntimeError('Latents must be passed')

        if (latent is not None) and (latents is not None):
            raise RuntimeError('Only latent or latents must be passed')

        if latent:
            # Standardize input to ease implementation
            latents = [latent]

        if prior:
            self._num_latents = len(latents)
        else:
            self._num_latents = latent.num_outputs


        self._latents_arr = ensure_module_list(latents)
        self._num_outputs = self.num_latents
        self._output_dim = self.num_latents

    def forward(self, x):
        """Compute f=T(x)."""
        return x

    def inverse(self, f):
        """Compute x=T^{-1}(f)."""
        return f

    def get_Z(self):
        Z_arr = batch_or_loop(
            lambda latent:  latent.sparsity.Z,
            [self.latents],
            [0],
            dim = self.num_latents,
            out_dim = 1,
            batch_type = get_batch_type(self.latents)
        )

        return Z_arr

    @property
    def latent_obj(self):
        # For consistency with other transform classes
        return self 

    @property
    def num_latents(self):
        return len(self.latents)


    def mean(self, X1: np.ndarray) -> np.ndarray:
        mean = batch_or_loop(
            lambda X1, latent:  latent.mean(X1),
            [X1, self.latents],
            [None, 0],
            dim = self.num_latents,
            out_dim = 1,
            batch_type = get_batch_type(self.latents)
        )

        mean = np.reshape(
            mean, 
            [self.num_outputs, X1.shape[0], 1]
        )

        return mean


    def covar(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        k_arr = batch_or_loop(
            lambda X1, X2, latent:  latent.covar(X1, X2),
            [X1, X2, self.latents],
            [None, None, 0],
            dim = self.num_latents,
            out_dim = 1,
            batch_type = get_batch_type(self.latents)
        )

        k_arr = np.reshape(
            k_arr, 
            [self.num_outputs, X1.shape[0], X2.shape[0]]
        )

        return k_arr


    def var(self, X1: np.ndarray) -> np.ndarray:
        var = batch_or_loop(
            lambda X1, latent:  latent.var(X1),
            [X1, self.latents],
            [None, 0],
            dim = self.num_latents,
            out_dim = 1,
            batch_type = get_batch_type(self.latents)
        )

        var = np.reshape(
            var, 
            [self.num_outputs, X1.shape[0]]
        )

        return var

    def full_var(self, X1: np.ndarray) -> np.ndarray:
        return self.covar(X1, X1)

    def state_space_representation(self, X_s):
        F_blocks, L_blocks, Qc_blocks, H_blocks, P_inf_blocks = batch_or_loop(
            lambda x_s, latent:  latent.kernel.to_ss(x_s),
            [X_s, self.latents],
            [None, 0],
            dim = self.num_latents,
            out_dim = 5,
            batch_type = get_batch_type(self.latents)
        )

        F = to_block_diag(F_blocks)
        L = to_block_diag(L_blocks)
        Qc = to_block_diag(Qc_blocks)
        P_inf = to_block_diag(P_inf_blocks)
        H = to_block_diag(H_blocks)

        return F, L, Qc, H, P_inf

    def expm(self, dt, X_s):
        A_blocks = batch_or_loop(
            lambda d, x_s, latent:  latent.kernel.expm(dt, x_s),
            [dt, X_s, self.latents],
            [None, None, 0],
            dim = self.num_latents,
            out_dim = 1,
            batch_type = get_batch_type(self.latents)
        )

        return to_block_diag(A_blocks)

class SumTransform(LinearTransform):
    def __init__(self, t1: Transform, t2: Transform):
        self.t1 = t1
        self.t2 = t2

        # Make sure t1 and t2 are compatable
        chex.assert_equal(
            self.t1.num_outputs,
            self.t2.num_outputs
        )

        self._num_outputs = self.t1.num_outputs
        self._num_latents = self.num_outputs

    def vec_mean(self, X1): 
        return self.t1.vec_mean(X1) + self.t2.vec_mean(X1)

    def mean(self, X1): 
        return self.t1.mean(X1) + self.t2.mean(X1)

    def covar(self, X1, X2): 
        return self.t1.covar(X1, X2) + self.t2.covar(X1, X2)

    def full_covar(self, X1, X2): 
        return self.t1.full_covar(X1, X2) + self.t2.full_covar(X1, X2)

    def var(self, X1): 
        return self.t1.var(X1) + self.t2.var(X1)

    def vec_var(self, X1): 
        return self.t1.vec_var(X1) + self.t2.vec_var(X1)

    def full_var(self, X1): 
        return self.t1.full_var(X1) + self.t2.full_var(X1)

class _One2One(Independent):
    def __init__(self, in_model: Transform, out_models: List[Transform]):
        self.in_model = in_model
        self.out_models = ensure_module_list(out_models)

        self._output_dim = self.in_model.output_dim
        self._input_dim = self.output_dim

    def mean(self, X1): 
        m =  self.in_model.mean(X1)
        chex.assert_shape(m, [self.output_dim, X1.shape[0], 1])
        return m

    def covar(self, X1, X2): 
        P = self.output_dim
        N1 = X1.shape[0]
        N2 = X2.shape[0]

        # precompute kernels from in_model
        prior_covar = self.in_model.covar(X1, X2)
        prior_var_1 = self.in_model.var(X1)
        prior_var_2 = self.in_model.var(X2)
        prior_mean_1 = self.in_model.mean(X1)
        prior_mean_2 = self.in_model.mean(X2)

        def _propogate(X1, X2, model_p, mean_p_1, mean_p_2, prior_var_1, prior_var_2, covar_p):
            return model_p.kernel.forward(X1, X2, mean_p_1, mean_p_2, prior_var_1, prior_var_2, covar_p)

        # push each outputs covar through a kernel
        covar = batch_or_loop(
            _propogate,
            [X1, X2, self.out_models, prior_mean_1, prior_mean_2, prior_var_1, prior_var_2, prior_covar],
            [None, None, 0, 0, 0, 0, 0, 0],
            dim=self.output_dim,
            out_dim=1,
            batch_type = BatchType.LOOP
        )

        chex.assert_shape(covar, [P, N1, N2])
        return covar

    def var(self, X1): 
        P = self.output_dim
        N1 = X1.shape[0]

        # precompute input mean and variances
        prior_var = self.in_model.var(X1)
        prior_mean_1 = self.in_model.mean(X1)

        def _propogate(X1, model_p, mean_p_1, prior_var_p):
            # TODO: model_p should just be a deep kernel
            return model_p.kernel.forward_diag(X1, mean_p_1, prior_var_p)

        # push each outputs covar through a kernel
        var = batch_or_loop(
            _propogate,
            [X1, self.out_models, prior_mean_1, prior_var],
            [None, 0, 0, 0, 0],
            dim=self.output_dim,
            out_dim=1,
            batch_type = BatchType.LOOP
        )

        chex.assert_shape(var, [P, N1])
        return var


class LatentSpecific(Transform):
    """ A transform that can only be applied to a single latent function """
    pass

class ElementWiseTransform(LatentSpecific):
    def __init__(self):
        super(ElementWiseTransform, self).__init__()
        self._num_latents = 1
        self._num_outputs = 1

class ParentPassThrough(Transform):
    """ Helper class to 'inherit' the parents methods """
    def var(self, XS: np.ndarray) -> np.ndarray :
        return self.parent.var(XS)

    def full_var(self, XS: np.ndarray) -> np.ndarray :
        return self.parent.full_var(XS)

class LinearOne2One(LinearTransform):
    def __init__(self, base_prior: 'Transform', transform_arr: list):
        self.base_prior = base_prior
        self.transform_arr = objax.ModuleList(transform_arr)
        self._output_dim = len(self.transform_arr)

    @property
    def latent_obj(self):
        return self.base_prior.latent_obj

    def forward(self, f):
        raise NotImplementedError()



class One2One(NonLinearTransform):
    def __init__(self, base_prior: 'Transform', transform_arr: list):
        self.base_prior = base_prior
        self.transform_arr = objax.ModuleList(transform_arr)
        self._output_dim = len(self.transform_arr)

    @property
    def latent_obj(self):
        return self.base_prior.latent_obj

    def forward(self, f):

        f_prop = self.base_prior.forward(f)

        num_outputs = len(self.transform_arr)

        f_prop = np.reshape(f_prop, [num_outputs, 1])

        f_transformed = batch_or_loop(
            lambda t_fn, f_p: t_fn.forward(f_p),
            [ self.transform_arr, f_prop ],
            [ 0, 0],
            dim = num_outputs,
            out_dim = 1,
            batch_type = get_batch_type(self.transform_arr)
        )


        f_transformed = np.reshape(f_transformed, [num_outputs, 1])

        return f_transformed



