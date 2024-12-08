import jax
import objax

import stgp
from stgp import settings
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern32, Matern52, ScaledMatern52, ScaledMatern32, SpatioTemporalSeperableKernel, WhiteNoiseKernel
from stgp.means.mean import FirstOrderDerivativeMean, SecondOrderDerivativeMean
from stgp.kernels.diff_op import FirstOrderDerivativeKernel, FirstOrderDerivativeKernel_2D, SecondOrderDerivativeKernel, SecondOrderOnlyDerivativeKernel
from stgp.likelihood import Gaussian, BlockDiagonalGaussian, ProductLikelihood
from stgp.models import GP
from stgp.transforms import OutputMap
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.data import Data
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import  progress_bar_callback
from stgp.kernels.spectral_mixture import SM_Component
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, FullConjugateGaussian, FullConjugatePrecisionGaussian
from stgp.transforms import Independent
from stgp.trainers.standard import VB_NG_ADAM, LBFGS, LikNoiseSplitTrainer, ADAM
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs, LTI_SDE, LTI_SDE_Full_State_Obs_With_Mask

import numpy as onp

def diff_gp(X, Y, time_diff = 1, space_diff = 1, base_kernel = None, fix_y=False, lik_var = 1.0):
    """
    Batch GP with a diff op kernel
    """

    if base_kernel is None:
        raise RuntimeError('Base Kernel must be passed!')

    N, P = Y.shape

    # construct time kernel
    if time_diff is not None:
        if time_diff == 1:
            kern = FirstOrderDerivativeKernel(base_kernel, input_index = 0)
        elif time_diff == 2:
            kern = SecondOrderDerivativeKernel(base_kernel, input_index = 0)

        elif time_diff == -2:
            kern = SecondOrderOnlyDerivativeKernel(base_kernel, input_index = 0)
    else:
        kern = base_kernel

    if space_diff is not None:
        if space_diff == 1:
            kern = FirstOrderDerivativeKernel(
                kern,
                input_index = 1,
                parent_output_dim = kern.output_dim
            )
        elif space_diff == 2:
            kern = SecondOrderDerivativeKernel(
                kern,
                input_index = 1,
                parent_output_dim = kern.output_dim
            )
        elif space_diff == -2:
            kern = SecondOrderOnlyDerivativeKernel(
                kern,
                input_index = 1,
                parent_output_dim = kern.output_dim
            )

    lik_arr = [Gaussian(lik_var) for p in range(P)]

    if fix_y:
        for lik in lik_arr:
            lik.fix()

    diff_op_prior = DifferentialOperatorJoint(
        GP(
            sparsity=stgp.sparsity.NoSparsity(Z=X), 
            kernel = base_kernel
        ),
        kernel = kern,
        is_base = True,
        has_parent=False
    )

    # Create Model
    m = stgp.models.GP(
        data = stgp.data.Data(X, Y),
        prior = diff_op_prior,
        likelihood = lik_arr,
    )

    return m


def diff_vgp(X, Y, time_diff = 1, space_diff = 1, diff_kern = None, base_kernel = None, fix_y=False, lik_arr = None, lik_var = 1.0, Z= None, Zs = None, ell_samples=None, prior_fn = None, whiten=False):

    if base_kernel is None:
        raise RuntimeError('Base Kernel must be passed!')

    N, P = Y.shape

    if diff_kern is None:
        # construct time kernel
        if time_diff is not None:
            if time_diff == 1:
                kern = FirstOrderDerivativeKernel(base_kernel, input_index = 0)
            elif time_diff == 2:
                kern = SecondOrderDerivativeKernel(base_kernel, input_index = 0)

            elif time_diff == -2:
                kern = SecondOrderOnlyDerivativeKernel(base_kernel, input_index = 0)
        else:
            kern = base_kernel

        if space_diff is not None:
            if space_diff == 1:
                kern = FirstOrderDerivativeKernel(
                    kern,
                    input_index = 1,
                    parent_output_dim = kern.output_dim
                )
            elif space_diff == 2:
                kern = SecondOrderDerivativeKernel(
                    kern,
                    input_index = 1,
                    parent_output_dim = kern.output_dim
                )
            elif space_diff == 2:
                kern = SecondOrderOnlyDerivativeKernel(
                    kern,
                    input_index = 1,
                    parent_output_dim = kern.output_dim
                )
    else:
        kern = diff_kern

    if type(lik_var) is not list:
        lik_var = [lik_var for p in range(P)]

    if lik_arr is None:
        if prior_fn is None:
            lik_arr = [Gaussian(lik_var[p]) for p in range(P)]
        else:
            lik_arr = [ProductLikelihood([Gaussian(lik_var[p])]) for p in range(P)]

    if fix_y:
        for lik in lik_arr:
            lik.fix()

    if (Z is None) and (Zs is None):
        sparsity = stgp.sparsity.NoSparsity(Z=X)
    elif Zs is not None:
        st_data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)
        sparsity = stgp.sparsity.SpatialSparsity(X_time = st_data.X_time, Z_space=Zs)
    else:
        sparsity = stgp.sparsity.FullSparsity(Z=Z)

    diff_op_prior = DifferentialOperatorJoint(
        GP(
            sparsity=sparsity, 
            kernel = base_kernel
        ),
        kernel = kern,
        is_base = True,
        has_parent=False,
        hierarchical=False
    )


    q = FullGaussianApproximatePosterior(dim = sparsity.Z.shape[0]*diff_op_prior.output_dim)

    if prior_fn is not None:
        # construct PDE transform
        diff_op_prior = prior_fn(diff_op_prior)

    # Create Model
    m = stgp.models.GP(
        data = stgp.data.Data(X, Y),
        prior = diff_op_prior,
        likelihood = lik_arr,
        inference='Variational',
        approximate_posterior=q,
        ell_samples = ell_samples,
        whiten=whiten
    )

    return m

def diff_hierarchical_vgp(X, Y, time_diff = 1, space_diff = 1, base_kernel = None, fix_y=False, lik_var = 1.0,  ell_samples=None, prior_fn = None, Z= None, Zs = None, whiten=False):


    if base_kernel is None:
        raise RuntimeError('Base Kernel must be passed!')

    N, P = Y.shape

    # construct time kernel
    if time_diff is not None:
        if time_diff == 1:
            time_kern = FirstOrderDerivativeKernel(base_kernel, input_index = 0)
            time_mean = FirstOrderDerivativeMean(parent_output_dim=1)
        elif time_diff == 2:
            time_kern = SecondOrderDerivativeKernel(base_kernel, input_index = 0)
            time_mean = SecondOrderDerivativeMean(parent_output_dim=1)

        elif time_diff == -2:
            time_kern = SecondOrderOnlyDerivativeKernel(base_kernel, input_index = 0)
            # TODO: is this ok?
            time_mean = SecondOrderDerivativeMean(parent_output_dim=1)

        time_output_dim = time_kern.output_dim
    else:
        time_output_dim = 1

    if space_diff is not None:
        if space_diff == 1:
            space_kern = FirstOrderDerivativeKernel( input_index = 1, parent_output_dim = time_output_dim)
            space_mean = FirstOrderDerivativeMean(parent_output_dim=time_output_dim, input_index=1)
        elif space_diff == 2:
            space_kern = SecondOrderDerivativeKernel( input_index = 1, parent_output_dim = time_output_dim)
            space_mean = SecondOrderDerivativeMean(parent_output_dim=time_output_dim, input_index=1)

        elif space_diff == -2:
            space_kern = SecondOrderOnlyDerivativeKernel( input_index = 1, parent_output_dim = time_output_dim)
            space_mean = SecondOrderDerivativeMean(parent_output_dim=time_output_dim, input_index=1)

    if prior_fn is None:
        lik_arr = [Gaussian(lik_var) for p in range(P)]
    else:
        lik_arr = [ProductLikelihood([Gaussian(lik_var)]) for p in range(P)]

    if fix_y:
        for lik in lik_arr:
            lik.fix()

    if (Z is None) and (Zs is None):
        sparsity = stgp.sparsity.NoSparsity(Z=X)
    elif Zs is not None:
        st_data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)
        sparsity = stgp.sparsity.SpatialSparsity(X_time = st_data.X_time, Z_space=Zs)
    else:
        sparsity = stgp.sparsity.FullSparsity(Z=Z)

    latent_gp = Independent([GP(
        sparsity=sparsity, 
        kernel = base_kernel
    )])

    if time_diff is not None:
        diff_op_prior_time = DifferentialOperatorJoint(
            latent_gp,
            # this should fail!!
            #kernel = FirstOrderDerivativeKernel(input_index = 0),
            #mean = FirstOrderDerivativeMean(parent_output_dim=1),
            kernel = time_kern,
            mean = time_mean,
            is_base = True,
            has_parent=True,
            hierarchical=True
        )
    else:
        diff_op_prior_time = latent_gp

    if space_diff is not None:
        # construct P(S | T)
        diff_op_prior = DifferentialOperatorJoint(
            diff_op_prior_time,
            kernel = space_kern,
            mean = space_mean,
            is_base = True,
            has_parent=True,
            hierarchical=True
        )
    else:
        diff_op_prior = diff_op_prior_time

    if prior_fn is not None:
        # construct PDE transform
        diff_op_prior = prior_fn(diff_op_prior)

    # only need to learn f
    q = FullGaussianApproximatePosterior(dim = sparsity.Z.shape[0] )

    # Create Model
    m = stgp.models.GP(
        data = stgp.data.Data(X, Y),
        prior = diff_op_prior,
        likelihood = lik_arr,
        inference='Variational',
        approximate_posterior=q,
        ell_samples = ell_samples,
        whiten=whiten
    )

    return m


def diff_hierarchical_sde_vgp(X, Y, time_diff = 1, space_diff = 1, time_kernel = None, space_kernel = None, fix_y=False, lik_var = 1.0, Z= None, ell_samples=None, prior_fn = None, keep_dims = None, parallel=False, verbose=False):
    if time_kernel is None:
        raise RuntimeError('Time Kernel must be passed!')

    include_space = not(space_kernel is None)

    if parallel == 'auto':
        if jax.devices()[0].device_kind == 'cpu':
            if verbose:
                print('running locally -- sequential filter used')
            parallel = False
        else:
            if verbose:
                print('running externally -- parallel filter used')
            parallel = True

    if parallel:
        filter_type = 'parallel'
    else:
        filter_type = 'sequential'

    N, P = Y.shape

    if include_space:
        data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)

        base_kernel = time_kernel * space_kernel

        base_sde_kernel = SpatioTemporalSeperableKernel(
            FirstOrderDerivativeKernel(time_kernel, input_index=0), 
            space_kernel
        )
    else:
        data = stgp.data.MultiOutputTemporalData(X=X, Y=Y, sort=True)

        base_kernel = time_kernel
        base_sde_kernel = base_kernel


    if type(lik_var) is not list:
        lik_var = [lik_var for p in range(P)]

    if prior_fn is None:
        lik_arr = [Gaussian(lik_var[p]) for p in range(P)]
    else:
        lik_arr = [ProductLikelihood([Gaussian(lik_var[p])]) for p in range(P)]

    if fix_y:
        for lik in lik_arr:
            lik.fix()

    # construct time kernel
    if time_diff is not None:
        if time_diff == 1:
            time_kern = FirstOrderDerivativeKernel(base_kernel, input_index = 0)
            time_mean = FirstOrderDerivativeMean(parent_output_dim=1)
        elif time_diff == 2:
            time_kern = SecondOrderDerivativeKernel(base_kernel, input_index = 0)
            time_mean = SecondOrderDerivativeMean(parent_output_dim=1)

        time_output_dim = time_kern.output_dim
    else:
        time_output_dim = 1


    diff_op_prior_time = DifferentialOperatorJoint(
        GP(
            sparsity=stgp.sparsity.NoSparsity(Z_ref=data._X), 
            kernel = base_kernel
        ),
        kernel = time_kern,
        mean = time_mean,
        is_base = True,
        has_parent=False,
        hierarchical=False
    )

    if include_space:

        if space_diff == 1:
            space_kern = FirstOrderDerivativeKernel( input_index = 1,)
            space_mean = FirstOrderDerivativeMean(input_index=1)
        elif space_diff == 2:
            space_kern = SecondOrderDerivativeKernel( input_index = 1,)
            space_mean = SecondOrderDerivativeMean(input_index=1)

        elif space_diff == -2:
            space_kern = SecondOrderOnlyDerivativeKernel( input_index = 1,)
            space_mean = SecondOrderDerivativeMean(input_index=1)

        # construct P(S | T)
        diff_op_prior = DifferentialOperatorJoint(
            diff_op_prior_time,
            kernel = space_kern,
            mean = space_mean,
            is_base = True,
            has_parent=True,
            hierarchical=True
        )
    else:
        diff_op_prior = diff_op_prior_time

    # surrogate model prior
    latent_sde_gp = GP(
        sparsity=stgp.sparsity.NoSparsity(Z=X), 
        kernel = base_sde_kernel
    )

    latent_sde_gp = Independent([latent_sde_gp])

    if keep_dims is None:
        latent_sde_gp = LTI_SDE_Full_State_Obs(latent_sde_gp)
    else:
        latent_sde_gp = LTI_SDE_Full_State_Obs_With_Mask(latent_sde_gp, keep_dims=keep_dims)


    if include_space:
        Q = diff_op_prior_time.output_dim
        B = data.Ns * Q
        q = FullConjugateGaussian(
            X = data._X,
            num_latents =  Q,
            block_size= B,
            num_blocks = data.Nt,
            surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                # in state-space format
                data = stgp.data.SpatioTemporalData(X=X, Y=onp.reshape(Y, [data.Nt,  Q, data.Ns]), sort=False, train_y=False), # we need gradients Y so set to be trainable
                likelihood=likelihood, 
                prior=latent_sde_gp,
                inference='Sequential',
                full_state_observed = True,
                filter_type=filter_type
            )
        )
    else:
        Q = diff_op_prior_time.output_dim
        B = data.Ns * Q
        q = FullConjugateGaussian(
            X = data._X,
            num_latents =  Q,
            block_size= B,
            num_blocks = data.Nt,
            surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                # in state-space format
                data = stgp.data.MultiOutputTemporalData(X=X, Y=onp.reshape(Y, [data.Nt,  Q, data.Ns]), sort=False, train_y=False), # we need gradients Y so set to be trainable
                likelihood=likelihood, 
                prior=latent_sde_gp,
                inference='Sequential',
                full_state_observed = True,
                filter_type=filter_type
            )
        )

    if prior_fn is not None:
        # construct PDE transform
        diff_op_prior = prior_fn(diff_op_prior)

    # Create Model
    m = stgp.models.GP(
        data = data,
        prior = diff_op_prior,
        likelihood = lik_arr,
        inference='Variational',
        approximate_posterior=q,
        ell_samples=ell_samples
    )

    return m

def diff_hierarchical_sparse_sde_vgp(X, Y, time_diff = 1, space_diff = 1, time_kernel = None, space_kernel = None, fix_y=False, lik_var = 1.0, Z= None, train_Z = True, ell_samples=None, prior_fn = None, keep_dims=None, whiten_space = False, parallel=False, precision=True, verbose=False):
    if time_kernel is None:
        raise RuntimeError('Time Kernel must be passed!')


    if Z is None:
        raise RuntimeError('Z must be passed!')

    if parallel == 'auto':
        if jax.devices()[0].device_kind == 'cpu':
            if verbose:
                print('running locally -- sequential filter used')
            parallel = False
        else:
            if verbose:
                print('running externally -- parallel filter used')
            parallel = True


    if parallel:
        filter_type = 'parallel'
    else:
        filter_type = 'sequential'

    include_space = not(space_kernel is None)

    N, P = Y.shape
    Ms = Z.shape[0]


    if include_space:
        data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)

        base_kernel = time_kernel * space_kernel

        base_sde_kernel = SpatioTemporalSeperableKernel(
            FirstOrderDerivativeKernel(time_kernel, input_index=0), 
            space_kernel,
            whiten_space = whiten_space
        )
    else:
        data = stgp.data.MultiOutputTemporalData(X=X, Y=Y, sort=True)

        base_kernel = time_kernel
        base_sde_kernel = base_kernel

    if type(lik_var) is not list:
        lik_var = [lik_var for i in range(P)]

    if prior_fn is None:
        lik_arr = [Gaussian(lik_var[p]) for p in range(P)]
    else:
        lik_arr = [ProductLikelihood([Gaussian(lik_var[p])]) for p in range(P)]

    if fix_y:
        for lik in lik_arr:
            lik.fix()

    # construct time kernel
    if time_diff is not None:
        if time_diff == 1:
            time_kern = FirstOrderDerivativeKernel(base_kernel, input_index = 0)
            time_mean = FirstOrderDerivativeMean(parent_output_dim=1)
        elif time_diff == 2:
            time_kern = SecondOrderDerivativeKernel(base_kernel, input_index = 0)
            time_mean = SecondOrderDerivativeMean(parent_output_dim=1)

        time_output_dim = time_kern.output_dim
    else:
        time_output_dim = 1


    Z_sparsity = stgp.sparsity.SpatialSparsity(data.X_time, Z, train=train_Z)

    diff_op_prior_time = DifferentialOperatorJoint(
        GP(
            sparsity=Z_sparsity, 
            kernel = base_kernel
        ),
        kernel = time_kern,
        mean = time_mean,
        is_base = True,
        has_parent=False,
        hierarchical=False
    )

    if include_space:

        if space_diff == 1:
            space_kern = FirstOrderDerivativeKernel( input_index = 1,)
            space_mean = FirstOrderDerivativeMean(input_index=1)
        elif space_diff == 2:
            space_kern = SecondOrderDerivativeKernel( input_index = 1)
            space_mean = SecondOrderDerivativeMean(input_index=1)
        elif space_diff == -2:
            space_kern = SecondOrderOnlyDerivativeKernel(input_index = 1)
            space_mean = SecondOrderOnlyDerivativeKernel(input_index=1)

        # construct P(S | T)
        diff_op_prior = DifferentialOperatorJoint(
            diff_op_prior_time,
            kernel = space_kern,
            mean = space_mean,
            is_base = True,
            has_parent=True,
            hierarchical=True,
            whiten_space = whiten_space
        )
    else:
        diff_op_prior = diff_op_prior_time

    # surrogate model prior
    latent_sde_gp = GP(
        sparsity=Z_sparsity, 
        kernel = base_sde_kernel
    )

    latent_sde_gp = Independent([latent_sde_gp], whiten_space = whiten_space)

    if keep_dims is None:
        latent_sde_gp = LTI_SDE_Full_State_Obs(latent_sde_gp, whiten_space = whiten_space)
    else:
        latent_sde_gp = LTI_SDE_Full_State_Obs_With_Mask(latent_sde_gp, keep_dims=keep_dims, whiten_space = whiten_space)


    Q = diff_op_prior_time.output_dim
    B = Ms * Q

    if precision:
        q_cls = FullConjugatePrecisionGaussian
    else:
        q_cls = FullConjugateGaussian

    q = q_cls(
        X = Z_sparsity,
        num_latents =  Q,
        block_size= B,
        num_blocks = data.Nt,
        surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
            # in state-space format
            data = stgp.data.SpatioTemporalData(X=X.raw_Z, Y=onp.reshape(Y, [data.Nt,  Q, Ms]), sort=False, train_y=True), # we need gradients Y so set to be trainable
            likelihood=likelihood, 
            prior=latent_sde_gp,
            inference='Sequential',
            full_state_observed = True,
            filter_type=filter_type
        )
    )

    # cannot train Z...

    if prior_fn is not None:
        # construct PDE transform
        diff_op_prior = prior_fn(diff_op_prior)

    # Create Model
    m = stgp.models.GP(
        data = data,
        prior = diff_op_prior,
        likelihood = lik_arr,
        inference='Variational',
        approximate_posterior=q,
        ell_samples=ell_samples
    )

    return m

def diff_sde_vgp(X, Y, time_diff = 1, space_diff = 1, time_kernel = None, space_kernel = None, fix_y=False, lik_var = 1.0,  ell_samples=None, prior_fn = None, keep_dims=None, parallel=False, verbose=False):

    if time_kernel is None:
        raise RuntimeError('Time Kernel must be passed!')

    if parallel == 'auto':
        if jax.devices()[0].device_kind == 'cpu':
            if verbose:
                print('running locally -- sequential filter used')
            parallel = False
        else:
            if verbose:
                print('running externally -- parallel filter used')
            parallel = True

    if parallel:
        filter_type = 'parallel'
    else:
        filter_type = 'sequential'


    include_space = not(space_kernel is None)

    N, P = Y.shape

    if include_space:
        data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)
        base_kernel = time_kernel * space_kernel
    else:
        data = stgp.data.MultiOutputTemporalData(X=X, Y=Y, sort=True)
        base_kernel = time_kernel

    Ms = data.Ns

    if type(lik_var) is not list:
        lik_var = [lik_var for p in range(P)]

    if prior_fn is None:
        lik_arr = [Gaussian(lik_var[p]) for p in range(P)]
    else:
        lik_arr = [ProductLikelihood([Gaussian(lik_var[p])]) for p in range(P)]

    if fix_y:
        for lik in lik_arr:
            lik.fix()

    # construct time kernel
    if time_diff is not None:
        if time_diff == 1:
            # for SDE model
            time_kern = FirstOrderDerivativeKernel(time_kernel, input_index = 0)
            time_mean = FirstOrderDerivativeMean(parent_output_dim=1)

            base_time_kern = FirstOrderDerivativeKernel(base_kernel, input_index = 0)
        elif time_diff == 2:
            time_kern = SecondOrderDerivativeKernel(time_kernel, input_index = 0)
            time_mean = SecondOrderDerivativeMean(parent_output_dim=1)

        time_output_dim = time_kern.output_dim
    else:
        time_output_dim = 1


    Z_sparsity = stgp.sparsity.NoSparsity(Z=X)

    diff_op_prior_time = DifferentialOperatorJoint(
        GP(
            sparsity=Z_sparsity, 
            kernel = base_kernel
        ),
        kernel = time_kern,
        mean = time_mean,
        is_base = True,
        has_parent=False,
        hierarchical=False
    )


    if include_space:
        if space_diff == 1:
            # for surrogate SDE
            space_kern = FirstOrderDerivativeKernel(space_kernel, input_index = 1)
            space_mean = FirstOrderDerivativeMean(input_index=1)

            # for vi model
            base_space_kern = FirstOrderDerivativeKernel(base_time_kern, input_index = 1, parent_output_dim=base_time_kern.output_dim)
        elif space_diff == 2:
            space_kern = SecondOrderDerivativeKernel(space_kernel, input_index = 1)
            space_mean = SecondOrderDerivativeMean(input_index=1)
        elif space_diff - 2:
            space_kern = SecondOrderOnlyDerivativeKernel(space_kernel, input_index = 1)
            space_mean = SecondOrderOnlyDerivativeKernel(input_index=1)

        base_sde_kernel = SpatioTemporalSeperableKernel(
            time_kern, 
            space_kern,
            spatial_output_dim = space_kern.output_dim
        )

        # construct P(S | T)
        diff_op_prior = DifferentialOperatorJoint(
            diff_op_prior_time,
            kernel = base_space_kern,
            mean = space_mean,
            is_base = True,
            has_parent=True,
            hierarchical=False
        )
    else:
        diff_op_prior = diff_op_prior_time
        base_sde_kernel = base_kernel


    # surrogate model prior
    latent_sde_gp = GP(
        sparsity=Z_sparsity, 
        kernel = base_sde_kernel
    )

    latent_sde_gp = Independent([latent_sde_gp])

    if keep_dims is None:
        latent_sde_gp = LTI_SDE_Full_State_Obs(latent_sde_gp)
    else:
        latent_sde_gp = LTI_SDE_Full_State_Obs_With_Mask(latent_sde_gp, keep_dims=keep_dims)


    def get_data(X, Y):
        # we need gradients Y so set to be trainable
        if include_space:
            return stgp.data.SpatioTemporalData(X=X, Y=onp.reshape(Y, [data.Nt,  Q, Ms]), sort=False, train_y=True) 
        return stgp.data.MultiOutputTemporalData(X=X, Y=onp.reshape(Y, [data.Nt,  Q, Ms]), sort=False, train_y=True)

    Q = diff_op_prior.output_dim
    B = Ms * Q
    q = FullConjugateGaussian(
        X = data._X,
        num_latents =  Q,
        block_size= B,
        num_blocks = data.Nt,
        surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
            # in state-space format
            data = get_data(X, Y),
            likelihood=likelihood, 
            prior=latent_sde_gp,
            inference='Sequential',
            full_state_observed = True,
            filter_type=filter_type
        )
    )

    # cannot train Z...

    if prior_fn is not None:
        # construct PDE transform
        diff_op_prior = prior_fn(diff_op_prior)

    # Create Model
    m = stgp.models.GP(
        data = data,
        prior = diff_op_prior,
        likelihood = lik_arr,
        inference='Variational',
        approximate_posterior=q,
        ell_samples=ell_samples
    )

    return m



