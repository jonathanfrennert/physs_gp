"""
We use the convention that a negative diff means to compute onl that corresponding derivative:
    ie a diff of 2 computes [f, df/dt, df2/dt^2] where as -2 only computes [f, df2/dt^2]
"""
import jax
import jax.numpy as np
import objax

import stgp
from stgp import settings
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern32, Matern52, ScaledMatern52, ScaledMatern32, SpatioTemporalSeperableKernel
from stgp.means.mean import FirstOrderDerivativeMean, SecondOrderDerivativeMean
from stgp.kernels.diff_op import FirstOrderDerivativeKernel, FirstOrderDerivativeKernel_2D, SecondOrderDerivativeKernel, SecondOrderOnlyDerivativeKernel, DummyDerivativeKernel, SecondOrderOnlyDerivativeKernel_2D, SecondOrderSpaceFirstOrderTimeDerivativeKernel_3D, FirstOrderDerivativeKernel_3D
from stgp.likelihood import Gaussian, BlockDiagonalGaussian, ProductLikelihood
from stgp.models import GP
from stgp.transforms import OutputMap
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.data import Data, SpatioTemporalData, TemporallyGroupedData
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import  progress_bar_callback
from stgp.kernels.spectral_mixture import SM_Component
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, FullConjugateGaussian, MeanFieldConjugateGaussian, MeanFieldApproximatePosterior
from stgp.transforms import Independent
from stgp.trainers.standard import VB_NG_ADAM, LBFGS, LikNoiseSplitTrainer, ADAM
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs, LTI_SDE, LTI_SDE_Full_State_Obs_With_Mask

import numpy as onp

import warnings

def _get_time_diff_kernel_mean(time_kernel, time_diff):
    if time_diff == 1:
        time_kern = FirstOrderDerivativeKernel(time_kernel, input_index = 0)
        time_mean = FirstOrderDerivativeMean(parent_output_dim=1)
    elif time_diff == 2:
        time_kern = SecondOrderDerivativeKernel(time_kernel, input_index = 0)
        time_mean = SecondOrderDerivativeMean(parent_output_dim=1)
    elif time_diff == -2:
        time_kern = SecondOrderOnlyDerivativeKernel(time_kernel, input_index = 0)
        time_mean = SecondOrderOnlyDerivativeKernel(parent_output_dim=1) # not used

    return time_kern, time_mean

def get_time_diff_kernel_mean(time_kernel, time_diff):
    Q = len(time_kernel)
    res = [
        _get_time_diff_kernel_mean(time_kernel[q], time_diff)
        for q in range(Q)
    ]
    
    return [res[q][0] for q in range(Q)], [res[q][1] for q in range(Q)]


def _get_space_diff_kernel_mean(space_kernel, space_diff, time_diff_kern = None, dim=1):
    """"
    There are two situations when creating a spatial diff kernel:
        
    Composition:
        The time kernel is passed to the space kernel and both the time and space outputs are computed together

    Heirarchical:
        The time kernel is not passed as the spatial part is computed separtely from the temporal part
    """
    if space_diff == None:
        space_mean = None
        space_kern = None
    elif space_diff == 1:
        if dim == 2:
            if time_diff_kern is None:
                # for surrogate SDE
                space_kern = FirstOrderDerivativeKernel(space_kernel, input_index = 1)
                space_mean = FirstOrderDerivativeMean(input_index=1)
            else:
                # for vi model
                space_kern = FirstOrderDerivativeKernel(time_diff_kern, input_index = 1, parent_output_dim=time_diff_kern.output_dim)
                space_mean = FirstOrderDerivativeMean(input_index=1, parent_output_dim=time_diff_kern.output_dim)
        else:
            if dim != 3: raise NotImplementedError()

            if time_diff_kern is None:
                # for surrogate SDE
                space_kern = FirstOrderDerivativeKernel_2D(space_kernel, input_index=1)
                space_mean = FirstOrderDerivativeMean(input_index=1) # not used
            else:
                # TODO: check this
                space_kern = FirstOrderDerivativeKernel_2D(space_kernel, input_index=1)
                space_mean = FirstOrderDerivativeMean(input_index=1) # not used


    elif space_diff == 2:
        if dim == 2:
            if time_diff_kern is None:
                space_kern = SecondOrderDerivativeKernel(space_kernel, input_index = 1)
                space_mean = SecondOrderDerivativeMean(input_index=1)
            else:
                space_kern = SecondOrderDerivativeKernel(time_diff_kern, input_index = 1, parent_output_dim=time_diff_kern.output_dim)
                space_mean = SecondOrderDerivativeMean(time_diff_kern, input_index = 1, parent_output_dim=time_diff_kern.output_dim)
        else:
            raise NotImplementedError()

    elif space_diff == -2:

        if dim == 2:
            if time_diff_kern is None:
                space_kern = SecondOrderOnlyDerivativeKernel(space_kernel, input_index = 1)
                space_mean = SecondOrderDerivativeMean(input_index=1) # not used
            else:
                space_kern = SecondOrderOnlyDerivativeKernel(time_diff_kern, input_index = 1, parent_output_dim=time_diff_kern.output_dim)
                space_mean = SecondOrderDerivativeMean(time_diff_kern, input_index = 1, parent_output_dim=time_diff_kern.output_dim) # not used
        else:
            if dim != 3: raise NotImplementedError()

            if time_diff_kern is None:
                space_kern = SecondOrderOnlyDerivativeKernel_2D(space_kernel, input_index = 1)
                space_mean = SecondOrderDerivativeMean(input_index=1)
            else:
                space_kern = SecondOrderOnlyDerivativeKernel_2D(time_diff_kern, input_index = 1, parent_output_dim=time_diff_kern.output_dim)
                space_mean = SecondOrderDerivativeMean(input_index=1, parent_output_dim=time_diff_kern.output_dim) # not used?

    return space_kern, space_mean

def get_space_diff_kernel_mean(space_kernel, space_diff, time_diff_kern = None, dim=2):
    Q = len(space_kernel)

    if time_diff_kern is None:
        time_diff_kern = [None for q in range(Q)]

    res = [
        _get_space_diff_kernel_mean(space_kernel[q], space_diff, time_diff_kern[q], dim=dim)
        for q in range(Q)
    ]
    
    return [res[q][0] for q in range(Q)], [res[q][1] for q in range(Q)]

def diff_gp(
    X, Y, num_latents=None, time_diff = 1, space_diff = 1, time_kernel = None, space_kernel = None, space_diff_kernel = None, fix_y=False, lik_arr=None, lik_var = 1.0, prior_fn = None, keep_dims=None , parallel = False, multioutput_prior = False, verbose=False,  meanfield=False, whiten=False, inference=None, hessian=False
):
    """
    Batch GP with derivative observations

    Args:
        num_latents [None | int] - Number of latent multi-variate GPs
        time_diff: [int] - number of temporal diffs to compute - will be the same across all latents
        space_diff: [int] - number of spatial diffs to compute - will be the same across all latents
        time_kernel: [list[kernel]|kernel]
        space_kernel: [list[kernel]|kernel]
        space_diff_kernel: Optional[]  - optional pre-computed space diff kernel. Useful for passing a closed form. Only works for hierarchial
        lik_var: [float|list[float]] - Gaussian likelihood noise. If a list if not passed the same value is initialised acrossed all outputs
        prior_fn: [None|callable] - Optional function to transform the prior with
        keep_dims: [None, list[int]] - Optional dims of the state-space state to observe. Useful when using higher order states corresponding to Matern52/72 etc.
        multioutput_prior[bool] - when multioutput the likelihood must be constructed as a list of productlikelihoods
        inference: [None, str] - None = batch
        hessian[bool] - If true compute df^2/dtds
        temporally_grouped[bool] - If true will use a TemporallyGrouped Data otherwise will construct SpatioTemporal
    """

    # Figure out what setting we are constructing a model in
    dim = X.shape[1]
    P = Y.shape[1]

    if dim > 1:
        include_space = True
    else:
        include_space = False

    if num_latents is None:
        if type(time_kernel) is list:
            num_latents = len(time_kernel)
        else:
            num_latents = 1

    # Convert to multi-latent form
    if type(time_kernel) is not list:
        time_kernel = [time_kernel]
        space_kernel = [space_kernel]
    
    # set up prior
    sparsity = stgp.sparsity.NoSparsity(Z=X)

    # Setup Prior
    if include_space:
        base_kernel = [
            time_kernel[q] * space_kernel[q]
            for q in range(num_latents)
        ]
    else:
         base_kernel = time_kernel


    #check for spatial cases where we can use more efficient kernel constructions
    if include_space and (dim == 2) and (time_diff == 1) and (space_diff == 1) and (not hessian):
        if verbose:
            print('Constructing 2D first order diff kernel')

        diff_op_prior = Independent([
            DifferentialOperatorJoint(
                GP(
                    sparsity=sparsity, 
                    kernel = base_kernel[i]
                ),
                kernel = FirstOrderDerivativeKernel_2D(base_kernel[i]),
                is_base = True,
                has_parent = False
            )
            for i in range(num_latents)
        ])
    elif include_space and (dim == 3) and (time_diff == 1) and (space_diff == 1) and (not hessian):
        if verbose:
            print('Constructing 3D first order diff kernel')

        diff_op_prior = Independent([
            DifferentialOperatorJoint(
                GP(
                    sparsity=sparsity, 
                    kernel = base_kernel[i]
                ),
                kernel = FirstOrderDerivativeKernel_3D(base_kernel[i]),
                is_base = True,
                has_parent = False
            )
            for i in range(num_latents)
        ])
    elif include_space and (dim == 3) and (time_diff == 1) and (space_diff == -2) and (not hessian):
        if verbose:
            print('Constructing 3D first order time and second order space kernel')

        diff_op_prior = Independent([
            DifferentialOperatorJoint(
                GP(
                    sparsity=sparsity, 
                    kernel = base_kernel[i]
                ),
                kernel = SecondOrderSpaceFirstOrderTimeDerivativeKernel_3D(base_kernel[i]),
                is_base = True,
                has_parent = False
            )
            for i in range(num_latents)
        ])
    else:
        print('Constructing composite order diff kernel')
        #no special cases available use composite construction
        #TODO: i think this is slower as required taking jacobians/hessians through each other

        # construct time kernel
        if time_diff is not None:
            composite_time_diff_kern, composite_time_diff_mean = get_time_diff_kernel_mean(base_kernel, time_diff)
        else:
            composite_time_diff_kern = [DummyDerivativeKernel(base_kernel[q]) for q in range(num_latents)]
            composite_time_diff_mean = [None]

        if include_space:
            # construct space kernel
            # we do not pass time as we are using a kalman filter which computes the spatial and temporal kernels separetely
            # in the composite case we  pass through the time_diff_kernel as want to compute something like
            #    kernel = FirstOrderDerivativeKernel(
            #        FirstOrderDerivativeKernel(base_kerns[i], input_index=0), 
            #        input_index=1, parent_output_dim = 2
            #    )
            composite_space_diff_kern, composite_space_diff_mean = get_space_diff_kernel_mean(composite_time_diff_kern, space_diff, composite_time_diff_kern, dim=dim)
            kern = composite_space_diff_kern
        else:
            kern = composite_time_diff_kern



        diff_op_prior = Independent([
            DifferentialOperatorJoint(
                GP(
                    sparsity=sparsity, 
                    kernel = base_kernel[q]
                ),
                kernel = kern[q],
                is_base = True,
                has_parent=False
            )
            for q in range(num_latents)
        ])

   # Setup likelihood
    if type(lik_var) is not list:
        lik_var = [lik_var for p in range(P)]

    # Setup Prior Transform
    if prior_fn is not None:
        # construct PDE transform
        diff_op_prior = prior_fn(diff_op_prior)

    if lik_arr is None:
        if multioutput_prior:
            lik_arr = [ProductLikelihood([Gaussian(lik_var[p])]) for p in range(P)]
        else:
            lik_arr = [Gaussian(lik_var[p]) for p in range(P)]

    if fix_y:
        for lik in lik_arr:
            lik.fix()

    # if a variational model set up the approximate posterior

    if inference == 'Variational':
        if meanfield:
            print('Setting a meanfield approximate posterior')
            q = MeanFieldApproximatePosterior(
                approximate_posteriors = [
                    FullGaussianApproximatePosterior(dim = sparsity.Z.shape[0]*diff_op_prior.base_prior.parent[q].output_dim)
                    for q in range(num_latents)
                ]
            )
        else:
            print('Setting a full Gaussian approximate posterior')
            q = FullGaussianApproximatePosterior(dim = sparsity.Z.shape[0]*diff_op_prior.base_prior.output_dim)

        # Create Model
        m = stgp.models.GP(
            data = stgp.data.Data(X, Y),
            prior = diff_op_prior,
            likelihood = lik_arr,
            inference = inference,
            approximate_posterior = q,
            whiten = whiten
        )
    else:
        # Create Model
        m = stgp.models.GP(
            data = stgp.data.Data(X, Y),
            prior = diff_op_prior,
            likelihood = lik_arr
        )


    return m


def diff_cvi_sde_vgp(
    X, Y,  num_latents=None, time_diff = 1, space_diff = 1, time_kernel = None, space_kernel = None, space_diff_kernel = None, fix_y=False,  lik_var = 1.0, Zs= None, train_Z = True, ell_samples=None, prior_fn = None, keep_dims=None , hierarchical=None, meanfield=False, parallel = False, multioutput_prior = False, temporally_grouped=False, lik_arr=None, verbose=False , overwrite_H=True, permute_H=True, minibatch_size = None, sde_prior = None, latent_sde_fn = None, stationary=True
):
    """
    Args:
        num_latents [None | int] - Number of latent multi-variate GPs
        time_diff: [int] - number of temporal diffs to compute - will be the same across all latents
        space_diff: [int] - number of spatial diffs to compute - will be the same across all latents
        time_kernel: [list[kernel]|kernel]
        space_kernel: [list[kernel]|kernel]
        space_diff_kernel: Optional[]  - optional pre-computed space diff kernel. Useful for passing a closed form. Only works for hierarchial
        lik_var: [float|list[float]] - Gaussian likelihood noise. If a list if not passed the same value is initialised acrossed all outputs
        Z_s: [None|np.ndarray|list[np.ndarray]] - Optional spatial inducing points. If passed then run in a sparse setting. 
        ell_samples: [None|int] - Optional number of monte-carlo samples to appoximate the ELL with
        prior_fn: [None|callable] - Optional function to transform the prior with
        keep_dims: [None, list[int]] - Optional dims of the state-space state to observe. Useful when using higher order states corresponding to Matern52/72 etc.
        hierarchical: [Optional[bool]] - Optional flag. If true then construct prior over temporal derivates and push spatial ones into the marginal/likelihood.
        meanfield: [bool] - default false. Construct a meanfield approximate posterior across the latents. Default is a full Gaussian.
        multioutput_prior[bool] - when multioutput the likelihood must be constructed as a list of productlikelihoods
        parallel: [bool, 'auto'] - default false. Whether or not use a parallel kalman filter and smoother.
        temporally_grouped: [bool] - Whether or not to use temporoally grouped or kronecker structure (only applied to ST problems)
    """

    if space_diff_kernel is not None:
        if not hierarchical:
            raise RuntimeError('Can only pass space_diff_kernel in a hierarchical model')

    # Figure out what setting we are constructing a model in
    dim = X.shape[1]

    if dim > 1:
        include_space = True
    else:
        include_space = False

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

    if num_latents is None:
        if type(time_kernel) is list:
            num_latents = len(time_kernel)
        else:
            num_latents = 1

    # Convert to multi-latent form
    if type(time_kernel) is not list:
        time_kernel = [time_kernel]
        space_kernel = [space_kernel]

    # Setup sequential data
    N, P = Y.shape
    if include_space:
        if temporally_grouped:
            data = TemporallyGroupedData(X=X, Y=Y, minibatch_size=minibatch_size)
        else:
            data = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True)
    else:
        data = stgp.data.MultiOutputTemporalData(X=X, Y=Y, sort=True)

    Ms = data.Ns

    # We only support the same inducing locations across all latent functions
    #   this is due to how the multi-latent kalman filter is constructed

    if Zs is not None:
        if not include_space:
            raise RuntimeError('Cannot have spatial inducing points in the 1D setting')

        if type(Zs) is list:
            raise RuntimeError('We only support the same inducing locations across latents')
        sparsity = stgp.sparsity.SpatialSparsity(data.X_time, Zs, train=train_Z)
        sparse = True
    else:
        # If Z not passed assume NoSparsity
        # Pass X from data so that it will be  properly sorted
        sparsity = stgp.sparsity.NoSparsity(Z_ref=data._X)
        sparse = False

    # Setup Prior
    if include_space:
        base_kernel = [
            time_kernel[q] * space_kernel[q]
            for q in range(num_latents)
        ]
    else:
         base_kernel = time_kernel

    # All models have time so we can directly construct the time prior

    # construct time kernel
    if time_diff is not None:
        time_diff_kern, time_diff_mean = get_time_diff_kernel_mean(time_kernel, time_diff)
        composite_time_diff_kern, composite_time_diff_mean = get_time_diff_kernel_mean(base_kernel, time_diff)
    else:
        # when no time kernel is passed we force keep dims to be [0] as we only observe f
        keep_dims = [0]
        time_diff_kern = [DummyDerivativeKernel(time_kernel[q]) for q in range(num_latents)]
        time_diff_mean = [None]
        composite_time_diff_kern = [DummyDerivativeKernel(base_kernel[q]) for q in range(num_latents)]
        composite_time_diff_mean = [None]

    if include_space:
        # construct space kernel
        # we do not pass time as we are using a kalman filter which computes the spatial and temporal kernels separetely
        if space_diff_kernel is None:
            space_diff_kern, space_diff_mean = get_space_diff_kernel_mean(space_kernel, space_diff, dim=dim)
        else:
            space_diff_kern = space_diff_kernel
            space_diff_mean = [None for i in range(space_diff)] # not used atm so just create nans

        # in the composite case we  pass through the time_diff_kernel as want to compute something like
        #    kernel = FirstOrderDerivativeKernel(
        #        FirstOrderDerivativeKernel(base_kerns[i], input_index=0), 
        #        input_index=1, parent_output_dim = 2
        #    )
        composite_space_diff_kern, composite_space_diff_mean = get_space_diff_kernel_mean(composite_time_diff_kern, space_diff, composite_time_diff_kern, dim=dim)


    # set up base prior
    if hierarchical or sparse:
        diff_op_prior_time = [
            DifferentialOperatorJoint(
                GP(
                    sparsity=sparsity, 
                    kernel = base_kernel[q]
                ),
                # we need to pass the composite kernel as internally this kernel is used to access the spatial kernel
                kernel = composite_time_diff_kern[q], 
                is_base = True,
                has_parent=False,
                hierarchical=False
            )
            for q in range(num_latents)
        ]

        if include_space and space_diff is not None:
            # construct P(S | T)
            diff_op_prior = [
                DifferentialOperatorJoint(
                    diff_op_prior_time[q],
                    kernel = space_diff_kern[q],
                    mean = None,
                    is_base = True,
                    has_parent=True,
                    hierarchical=hierarchical
                )
                for q in range(num_latents)
            ]
        else:
            diff_op_prior = diff_op_prior_time

    else:
        if include_space:
            diff_op_prior = [
                DifferentialOperatorJoint(
                    GP(
                        sparsity=sparsity, 
                        kernel = base_kernel[q]
                    ),
                    kernel = composite_space_diff_kern[q],
                    is_base = True,
                    has_parent = False
                )
                for q in range(num_latents)
            ]
        else:
            diff_op_prior = [
                DifferentialOperatorJoint(
                    GP(
                        sparsity=sparsity, 
                        kernel = base_kernel[q]
                    ),
                    kernel = composite_time_diff_kern[q],
                    is_base = True,
                    has_parent = False
                )
                for q in range(num_latents)
            ]

    diff_op_prior = Independent(diff_op_prior)

    # setup surrogate SDE prior

    if include_space:
        if hierarchical:
            # when hierachical we do not compute the spatial derivates using the filter
            base_st_kerns = [
                SpatioTemporalSeperableKernel(
                    time_diff_kern[q], 
                    space_kernel[q],
                    stationary=stationary
                )
                for q in range(num_latents)
            ]
        else:
            base_st_kerns = [
                SpatioTemporalSeperableKernel(
                    time_diff_kern[q], 
                    space_diff_kern[q],
                    spatial_output_dim = space_diff_kern[q].d_computed,
                    stationary=stationary
                )
                for q in range(num_latents)
            ]
    else:
        base_st_kerns = time_diff_kern

    # surrogate model prior
    if latent_sde_fn is not None:
        latent_sde_gp = latent_sde_fn(
            sparsity = sparsity, 
            base_st_kerns = base_st_kerns,
            num_latents = num_latents, 
            overwrite_H = overwrite_H, 
            permute_H = permute_H,
            keep_dims = keep_dims
        )

    else:
        if meanfield:
            if keep_dims is None:
                latent_sde_gp = Independent([
                    LTI_SDE_Full_State_Obs(
                        Independent([
                            GP(
                                sparsity=sparsity, 
                                kernel = base_st_kerns[i]
                            )
                        ]),
                        overwrite_H=overwrite_H,
                        permute=permute_H
                    )
                    for i in range(num_latents)
                ])
            else:
                latent_sde_gp = Independent([
                    LTI_SDE_Full_State_Obs_With_Mask(
                        Independent([
                            GP(
                                sparsity=sparsity, 
                                kernel = base_st_kerns[i]
                            )
                        ]),
                        keep_dims=keep_dims,
                        overwrite_H=overwrite_H,
                        permute=permute_H
                    )
                    for i in range(num_latents)
                ])

        else:
            if keep_dims is None:
                if sde_prior is None:
                    latent_sde_gp = LTI_SDE_Full_State_Obs(
                        Independent([
                            GP(
                                sparsity=sparsity, 
                                kernel = base_st_kerns[q]
                            )
                            for q in range(num_latents)
                        ]),
                        overwrite_H=overwrite_H,
                        permute=permute_H
                    )
                else:
                    latent_sde_gp = sde_prior(
                        Independent([
                            GP(
                                sparsity=sparsity, 
                                kernel = base_st_kerns[q]
                            )
                            for q in range(num_latents)
                        ])
                    )
            else:
                if sde_prior is None:
                    latent_sde_gp = LTI_SDE_Full_State_Obs_With_Mask(
                        Independent([
                            GP(
                                sparsity=sparsity, 
                                kernel = base_st_kerns[q]
                            )
                            for q in range(num_latents)
                        ]),
                        keep_dims=keep_dims,
                        overwrite_H=overwrite_H,
                        permute=permute_H
                    )
                else:
                    latent_sde_gp = sde_prior(
                        Independent([
                            GP(
                                sparsity=sparsity, 
                                kernel = base_st_kerns[q]
                            )
                            for q in range(num_latents)
                        ])
                    )




    # setup approximate posterior

    # When using keep_dims  only a subset of the full kalman state will be observed
    #    and the shape of Y and the likelihood only needs to be defined across the observed ones
    if keep_dims:
        state_dim = len(keep_dims)
    else:
        state_dim = time_kernel[0].state_space_dim()

    if include_space:
        if not hierarchical:
            # when not hierarchical 
            state_dim =  state_dim * space_diff_kern[0].output_dim

    if include_space:
        if sparse:
            Ms = sparsity.raw_Z.Ns
        else:
            Ms = data._X.Ns
    else:
        Ms = 1

    if meanfield:
        if include_space:
            if verbose:
                print(f'Q: {num_latents}, state_dim: {state_dim}, Nt: {data.Nt}, Ns: {data.Ns}, Ms: {Ms}')

            # ====== SPATIO-TEMPORAL MEANFIELD =======
            q = MeanFieldConjugateGaussian(
                approximate_posteriors = [
                    FullConjugateGaussian(
                        X = sparsity,
                        num_latents = state_dim,
                        block_size= Ms * state_dim,
                        num_blocks = data.Nt,
                        surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                            # in state-space format
                            data = stgp.data.SpatioTemporalData(X=sparsity.raw_Z, Y=np.reshape(Y, [data.Nt, state_dim, Ms]), sort=False),
                            likelihood=likelihood, 
                            prior=latent_sde_gp.parent[q],
                            inference='Sequential',
                            filter_type=filter_type,
                            full_state_observed=True
                        )
                    )
                    for q in range(num_latents)
                ]
            )
        else:
            # ====== TEMPORAL MEANFIELD =======
            q = MeanFieldConjugateGaussian(
                approximate_posteriors = [
                    FullConjugateGaussian(
                        X = sparsity,
                        num_latents = state_dim,
                        block_size= Ms * state_dim,
                        num_blocks = data.Nt,
                        surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                            # in state-space format
                            data = stgp.data.MultiOutputTemporalData(X=sparsity.raw_Z, Y=np.reshape(Y, [data.Nt, state_dim, Ms]), sort=False),
                            likelihood=likelihood, 
                            prior=latent_sde_gp.parent[q],
                            inference='Sequential',
                            filter_type=filter_type,
                            full_state_observed=True
                        )
                    )
                    for q in range(num_latents)
                ]
            )
    else:

        if include_space:
            if verbose:
                print(f'Q: {num_latents}, state_dim: {state_dim}, Nt: {data.Nt}, Ns: {data.Ns}, Ms: {Ms}')

            q = FullConjugateGaussian(
                X = sparsity,
                num_latents =  num_latents * state_dim,
                block_size= Ms * state_dim * num_latents,
                num_blocks = data.Nt,
                surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                    # in state-space format
                    data = stgp.data.SpatioTemporalData(X=sparsity.raw_Z, Y=np.reshape(Y, [data.Nt, state_dim * num_latents, Ms]), sort=False),
                    likelihood=likelihood, 
                    prior=latent_sde_gp,
                    inference='Sequential',
                    filter_type=filter_type,
                    full_state_observed=True
                )
            )
        else:
            q = FullConjugateGaussian(
                X = sparsity,
                num_latents =  num_latents * state_dim,
                block_size= Ms * state_dim * num_latents,
                num_blocks = data.Nt,
                surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
                    # in state-space format
                    data = stgp.data.MultiOutputTemporalData(X=sparsity.raw_Z, Y=np.reshape(Y, [data.Nt, state_dim * num_latents, Ms]), sort=False),
                    likelihood=likelihood, 
                    prior=latent_sde_gp,
                    inference='Sequential',
                    filter_type=filter_type,
                    full_state_observed=True
                )
            )

    # Setup Prior Transform
    if prior_fn is not None:
        # construct PDE transform
        diff_op_prior = prior_fn(diff_op_prior)

        if diff_op_prior is None:
            warnings.warn('prior_fn return None')



    if lik_arr is None:
        # Setup likelihood
        if type(lik_var) is not list:
            lik_var = [lik_var for p in range(P)]

        lik_arr = [Gaussian(lik_var[p]) for p in range(P)]

    if multioutput_prior:
        lik_arr = [ProductLikelihood([lik_arr[p]]) for p in range(P)]

    if fix_y:
        for lik in lik_arr:
            lik.fix()


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
