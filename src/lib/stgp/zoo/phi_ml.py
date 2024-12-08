import jax
import jax.numpy as np

from ..transforms.multi_output import LMC
from .sde_diff import diff_cvi_sde_vgp, diff_gp

def magnetic_field_strength_H(
    X, 
    Y,
    time_kernel = None,
    space_kernel = None,
    space_diff_kernel = None,
    hierarchical = False,
    Zs = None,
    lik_var = 1.0,
    fix_y = False,
    meanfield = False,
    verbose=True,
    keep_dims=None,
    model = None,
    parallel=False,
    temporally_grouped=False,
    include_potential_function=False,
    whiten=False
):
    assert X.shape[1] == 3

    if include_potential_function:
        assert Y.shape[1] == 4
    else:
        assert Y.shape[1] == 3


    def prior_fn(latents):
        if model == 'sde_cvi':
            # [f dx dy dt dtdx dtdy]
            if include_potential_function:
                W_curl_free = np.array([
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0],
                    [0, -1, 0, 0, 0, 0],
                    [0, 0, -1, 0, 0, 0],
                ])

                out_dim = 4
                in_dim = 6
            else:
                W_curl_free = np.array([
                    [0, 0, 0, -1, 0, 0],
                    [0, -1, 0, 0, 0, 0],
                    [0, 0, -1, 0, 0, 0],
                ])

                out_dim = 3
                in_dim = 6
        else:
            # [f dx dy dz]
            if include_potential_function:
                W_curl_free = np.array([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, -1],
                ])

                out_dim = 4
                in_dim = 4
            else:
                W_curl_free =  np.array([
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, -1],
                ])

                out_dim = 3
                in_dim = 4

        W = W_curl_free

        lmc_prior =  LMC(
            latents=latents,
            input_dim=in_dim,
            output_dim=out_dim,
            W = W
        )

        lmc_prior._W.fix()

        return lmc_prior

    if model == 'sde_cvi':
        return diff_cvi_sde_vgp(
            X, 
            Y,
            num_latents=1,
            time_diff = 1,
            space_diff = 1,
            time_kernel = time_kernel,
            space_kernel = space_kernel,
            space_diff_kernel = space_diff_kernel,
            hierarchical = hierarchical,
            Zs = Zs,
            lik_var = lik_var,
            fix_y = fix_y,
            meanfield = meanfield,
            prior_fn = prior_fn,
            verbose=verbose,
            keep_dims=keep_dims,
            parallel=parallel,
            temporally_grouped=temporally_grouped
        ) 
    elif model == 'vgp':
        return diff_gp(
            X, 
            Y,
            num_latents=1,
            time_diff = 1,
            space_diff = 1,
            time_kernel = time_kernel,
            space_kernel = space_kernel,
            lik_var = lik_var,
            fix_y = fix_y,
            meanfield = meanfield,
            prior_fn = prior_fn,
            verbose = verbose,
            inference='Variational',
            whiten=whiten
        )
    elif model == 'batch_gp':
        return diff_gp(
            X, 
            Y,
            num_latents=1,
            time_diff = 1,
            space_diff = 1,
            time_kernel = time_kernel,
            space_kernel = space_kernel,
            lik_var = lik_var,
            fix_y = fix_y,
            prior_fn = prior_fn,
            verbose = verbose
        )
    else:
        raise NotImplementedError()

def helmholtz_3D(
    X, 
    Y,
    time_kernel = None,
    space_kernel = None,
    space_diff_kernel = None,
    hierarchical = False,
    Zs = None,
    lik_var = 1.0,
    fix_y = False,
    meanfield = False,
    verbose=True,
    keep_dims=None,
    model = None,
    parallel=False,
    temporally_grouped=False,
    whiten=False,
    minibatch_size=None
):
    """

        In 3D we essentially construct 'independent' 2d helmholtz priors at each time step
        
        Args:
            model: [batch, sde_cvi]

    """

    if model is None:
        model = 'sde_cvi'


    assert X.shape[1] == 3

    def prior_fn(latents):
        
        # [f dx dy] [f dx dy] (e.g x = lat, y = lon)
        W_curl_free = np.array([
            [0, 1, 0, 0, 0, 1],  
            [0, 0, 1, 0, -1, 0],    
        ])

        out_dim = 2
        in_dim = 6

        W = W_curl_free

        lmc_prior =  LMC(
            latents=latents,
            input_dim=in_dim,
            output_dim=out_dim,
            W = W
        )

        lmc_prior._W.fix()

        return lmc_prior

    if model == 'sde_cvi':
        return diff_cvi_sde_vgp(
            X, 
            Y,
            num_latents=2,
            time_diff = None,
            space_diff = 1,
            time_kernel = time_kernel,
            space_kernel = space_kernel,
            space_diff_kernel = space_diff_kernel,
            hierarchical = hierarchical,
            Zs = Zs,
            lik_var = lik_var,
            fix_y = fix_y,
            meanfield = meanfield,
            prior_fn = prior_fn,
            verbose=verbose,
            keep_dims=keep_dims,
            parallel=parallel,
            temporally_grouped=temporally_grouped,
            minibatch_size=minibatch_size
        ) 
    elif model == 'vgp':

        if minibatch_size is not None:
            raise NotImplementedError()

        return diff_gp(
            X, 
            Y,
            num_latents=2,
            time_diff = None,
            space_diff = 1,
            time_kernel = time_kernel,
            space_kernel = space_kernel,
            lik_var = lik_var,
            fix_y = fix_y,
            meanfield = meanfield,
            prior_fn = prior_fn,
            verbose = verbose,
            inference='Variational',
            whiten=whiten
        )
    elif model == 'batch_gp':
        if minibatch_size is not None:
            raise NotImplementedError()

        return diff_gp(
            X, 
            Y,
            num_latents=2,
            time_diff = None,
            space_diff = 1,
            time_kernel = time_kernel,
            space_kernel = space_kernel,
            lik_var = lik_var,
            fix_y = fix_y,
            prior_fn = prior_fn,
            verbose = verbose
        )
    else:
        raise NotImplementedError()

def helmholtz(
    X, 
    Y,
    time_kernel = None,
    space_kernel = None,
    space_diff_kernel = None,
    hierarchical = False,
    Zs = None,
    lik_var = 1.0,
    fix_y = False,
    meanfield = False,
    verbose=True,
    keep_dims=None,
    model = None,
    parallel=False,
    temporally_grouped=False,
    whiten=False
):
    """
        The helmholtz (2d) decomposition is defined as in https://arxiv.org/pdf/2302.10364.pdf

        Indendent GPs are placed on the stream (psi) and potential (phi) functions

        The ocean flow is then
            flow=  grad(potential) + rot(stream)

        where
            grad(phi) = [d phi/dt, d phi/ds]
            rot(psi) = [d psi/ds, - d psi/dt]

        leading to 
            flow = [d phi/dt + d psi/ds, d phi/ds - d psi/dt]
        Args:
            model: [batch, sde_cvi]
    """

    if model is None:
        model = 'sde_cvi'


    # only defined for 2d methods atm
    assert X.shape[1] == 2

    def prior_fn_batch_efficient(latents):
        """ When using a 2D diff op prior """

        #Holmholtz prior

        W = np.array([
            # [f, ft, fs]_q
            [0, 1, 0,   0, 0, 1], # ft + fd
            [0, 0, 1,   0, -1, 0], # fs - ft
        ])
        out_dim = 2
        in_dim = 6

        lmc_prior =  LMC(
            latents=latents,
            input_dim=in_dim,
            output_dim=out_dim,
            W = W
        )

        lmc_prior._W.fix()

        return lmc_prior

    def prior_fn(latents):
        """
        The Helmholtz prior is defined as 
            [df1/dx df1/dy]^T + [-df2/dy df2/dx]^T
        """

        W = np.array([
            # [f, fs, ft, fts]_q
            [0, 0, 1, 0,   0, 1, 0, 0], # ft + fs
            [0, 1, 0, 0,   0, 0, -1, 0], # fs - ft
        ])
        out_dim = 2
        in_dim = 8

        lmc_prior =  LMC(
            latents=latents,
            input_dim=in_dim,
            output_dim=out_dim,
            W = W
        )

        lmc_prior._W.fix()

        return lmc_prior

    if model == 'sde_cvi':
        return diff_cvi_sde_vgp(
            X, 
            Y,
            num_latents=2,
            time_diff = 1,
            space_diff = 1,
            time_kernel = time_kernel,
            space_kernel = space_kernel,
            space_diff_kernel = space_diff_kernel,
            hierarchical = hierarchical,
            Zs = Zs,
            lik_var = lik_var,
            fix_y = fix_y,
            meanfield = meanfield,
            prior_fn = prior_fn,
            verbose=verbose,
            keep_dims=keep_dims,
            parallel=parallel,
            temporally_grouped=temporally_grouped
        ) 
    elif model == 'vgp':
        return diff_gp(
            X, 
            Y,
            num_latents=2,
            time_diff = 1,
            space_diff = 1,
            time_kernel = time_kernel,
            space_kernel = space_kernel,
            lik_var = lik_var,
            fix_y = fix_y,
            meanfield = meanfield,
            prior_fn = prior_fn_batch_efficient,
            verbose = verbose,
            inference='Variational',
            whiten=whiten
        )
    elif model == 'batch_gp':
        return diff_gp(
            X, 
            Y,
            num_latents=2,
            time_diff = 1,
            space_diff = 1,
            time_kernel = time_kernel,
            space_kernel = space_kernel,
            lik_var = lik_var,
            fix_y = fix_y,
            prior_fn = prior_fn_batch_efficient,
            verbose = verbose
        )
    else:
        raise NotImplementedError()


