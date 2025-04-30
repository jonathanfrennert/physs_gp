import sys
# Local Run 
sys.path.append('../../') 
# Cluster Run
sys.path.append('../') 

import objax
import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import sdem
from sdem import Experiment
from sdem.utils import read_yaml, get_all_permutations, print_dict, Split
from sdem.modelling.prediction import collect_results

import stdata
from stdata.prediction import batch_predict
from stdata.grids import create_spatial_grid, pad_with_nan_to_make_grid

import jax.numpy as jnp

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from rich.console import Console
from timeit import default_timer as timer

import stgp
from stgp.zoo.multi_output import gp_regression, lmc_regression, lmc_drd_regression, gprn_regression, gprn_drd_regression, gprn_drd_nv_regression
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.standard import VB_NG_ADAM, LBFGS, LikNoiseSplitTrainer, ADAM
from stgp import settings
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern52, ScaledMatern52, ScaledMatern72, SpatioTemporalSeperableKernel
from stgp.kernels.arccosine import ArcCosine, NeuralNetworkKernel
from stgp.kernels.wiener import IntegratedWiener
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D, SecondOrderDerivativeKernel_1D, SecondOrderDerivativeKernel_1D, SecondOrderSpaceFirstOrderTimeDerivativeKernel_2D, FirstOrderDerivativeKernel, SecondOrderDerivativeKernel, SecondOrderOnlyDerivativeKernel
from stgp.likelihood import Gaussian, PowerLikelihood, ProductLikelihood
from stgp.models import GP
from stgp.means.mean import SecondOrderDerivativeMean_2D, SecondOrderSpaceFirstOrderTimeDerivativeMean_2D, SecondOrderDerivativeMean, FirstOrderDerivativeMean, SecondOrderOnlyDerivativeMean
from stgp.transforms import LinearTransform, OutputMap, MultiOutput, DataLatentPermutation, Independent
from stgp.core.model_types import get_model_type
from stgp.transforms.pdes import DifferentialOperatorJoint, HeatEquation2D, Pendulum1D, DampedPendulum1D, AllenCahn, TaylorLinearizedDE
from stgp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian, FullGaussianApproximatePosterior, FullConjugatePrecisionGaussian
from stgp.computation.solvers.euler import euler
from stgp.kernels import SpatioTemporalSeperableKernel, Matern32, RBF, ScaledMatern72, ScaledMatern32
from stgp.transforms.sdes import LTI_SDE, LTI_SDE_Full_State_Obs, LinearizedFilter_SDE
from stgp.likelihood import ReshapedGaussian, DiagonalGaussian, BlockDiagonalGaussian
from stgp.kernels.diff_op import SecondOrderOnlyDerivativeKernel

from stgp.zoo.sde_diff import diff_cvi_sde_vgp

from setup_data import get_data_file_names

from experiment_utils import utils

# Create experiment
ex = Experiment(__file__)

# For pretty printing
console = Console()

@ex.configs
def get_config():
    num_epochs = 1

    options = {
        'fold': [0],
        'noise_fold': [0],
        'name': ['ac'],
        'sif': ['~/projects/deep_kernel/deep_kernel_stgp.sif'],
        'lib': ['stgp'],
        'model': ['pigp_ekf'],
        Split(): [
            {
                'M': [20],
                'lengthscale': [0.1]
            },
            {
                'M': [40],
                'lengthscale': [0.1]
            },
        ],
        'whiten': [False],
        'num_colocation': [[100, 10]],
        'ell_samples': [None],
        'adam_lr': [0.01],
        'ng_lr': [1.0],
        'train_Z': [False],
        'ng_schedule_iters': [None],
        'parallel': [False],
        'checkpoint': [False], # we need to run twice, with and without checkpoint, to get an accurate estimate of run time,
        'max_iters': [1],
        'ng_samples': [None],
        'seed': [0],
        'include_boundary': [False]
    }
    return  get_all_permutations(options)

def add_colocation_points(config, X, Y):
    X_colocation = create_spatial_grid(0.0, 1.0, -1.0, 1.0, config['num_colocation'][0], config['num_colocation'][1])

    Y_colocation = np.ones([X_colocation.shape[0], Y.shape[1]])*np.nan

    if X is None:
        X = X_colocation
        Y = Y_colocation
    else:
        X = np.vstack([X, X_colocation])
        Y = np.vstack([Y, Y_colocation])
    Y = np.hstack([Y, np.zeros_like(Y)])
    return X, Y

def get_model(config, X, Y):
    q = 1
    var = 1.0

    X, Y = add_colocation_points(config, X, Y)
    Y = Y[:, [0]]

    PDE_IN_PRIOR_ONLY = True

    def prior_fn(latents):

        # what is the ordering of the outputs here? and what ordering do we need?
        prior_outputs = OutputMap(
            latents,
            [
                [0], # x
                [0, 2, 1],  # [f, dt, ds2]
            ]
        )

        allen_prior = AllenCahn(
            prior_outputs[1], train=False , m_init_dim=39*4 # m_init not used, just need to pass something
        )
        lin_allen = OutputMap(
            TaylorLinearizedDE(
                prior_outputs[1],
                allen_prior,
                input_dim = 4,
                output_dim = 1,
                data_y_index = [1]
            ),
            [
                [0]
            ]
        )


        if PDE_IN_PRIOR_ONLY:
            return MultiOutput([prior_outputs[0]])
        prior = MultiOutput([prior_outputs[0]]+[lin_allen])
        return prior

    def latent_sde_fn(sparsity = None, base_st_kerns = None, num_latents = None, overwrite_H = None, permute_H = None, keep_dims = None):
        latent_gp = Independent([
            GP(
                sparsity = sparsity, 
                kernel =  base_st_kerns[i],
                prior = True
            )
            for i in range(1)
        ])

        xs = np.squeeze(sparsity.raw_Z.X_space)
        if config['include_boundary']:
            print('Including Boundary Conditions')
            xs = np.squeeze(sparsity.raw_Z.X_space)
            Nt = sparsity.raw_Z.Nt
            f_init =  jax.vmap(lambda x : (x**2)*jnp.cos(jnp.pi*x))(xs)
            f_init_ds = jax.vmap(lambda x: (2-(jnp.pi**2)*(x**2))*jnp.cos(jnp.pi*x)-4*jnp.pi*x*jnp.sin(jnp.pi*x))(xs)
            f_init_dt = f_init*0.0
            f_init_dtds = f_init*0.0

            # this is ds dt space -- need ds space dt
            m_init = np.hstack([f_init, f_init_dt, f_init_ds, f_init_dtds])
            #m_init = np.transpose(np.reshape(m_init, [2, 2, data.Ns]), [0, 2, 1]).reshape([1, -1]).astype(np.float64)
            m_init = m_init.reshape([1, -1]).astype(np.float64)

            boundary_conditions = m_init
            boundary_conditions = np.vstack([boundary_conditions, np.ones([Nt-1, boundary_conditions.shape[1]])*np.nan])[..., None]
        else:
            print('No Boundary Conditions')
            boundary_conditions = None

        return  AllenCahn(LTI_SDE_Full_State_Obs(latent_gp), train=False, m_init_dim=xs.shape[0]*4, boundary_conditions = boundary_conditions, train_m_init=False, boundary_by_init=False, observe_data=True)


    iwp = True
    if iwp:
        stationary=False
        base_kerns = [
            IntegratedWiener(q=1, variance=var, stable_state_covariance = 1e6, active_dims=[0])
        ]
    else:
        stationary=True

        base_kerns = [
            Matern32(input_dim=1, lengthscales=[0.05], active_dims=[0])
        ]

    space_kerns = [
        Matern52(input_dim=1, lengthscales=[config['lengthscale']], active_dims=[1])
    ]

    if PDE_IN_PRIOR_ONLY:
        lik_arr = [
            Gaussian(variance=1e-4)
        ]
    else:
        lik_arr = [
            Gaussian(variance=0.01),
            Gaussian(variance=1e-4)
        ]

    data = stgp.data.SpatioTemporalData(X=X, Y=Y)
    print('data: ', data.Nt, data.Ns)
    print('Z: ', config['M'])

    #Z = np.linspace(-1, 1, 10)[:, None]
    #Z = np.linspace(-1, 1, 10)[:, None]
    Z = np.linspace(-1, 1, config['M'])[:, None]

    m = diff_cvi_sde_vgp(
        X = X,
        Y = Y,
        Zs = Z,
        num_latents=1,
        time_diff=1,
        space_diff = -2,
        multioutput_prior=True,
        prior_fn=prior_fn,
        time_kernel=base_kerns,
        space_kernel=space_kerns,
        ell_samples=None,
        lik_arr=lik_arr,
        fix_y=True,
        parallel='auto',
        #keep_dims = [0, 1],
        permute_H=False,
        overwrite_H=True,
        meanfield=False,
        latent_sde_fn = latent_sde_fn,
        stationary=stationary
    )
    
    return m


@ex.model
def model(config, model_root: Path = '.', restore=True, checkpoint_epoch=None):
    model_root = Path(model_root)
    fold = config['fold']
    noise_fold = config['noise_fold']
    fnames = get_data_file_names(fold, noise_fold)

    # dont make folder structure if we are restoring as it should already exist
    data_root, results_root, checkpoint_folder = utils.get_and_ensure_folder_structure(mkdir = ~ restore)

    checkpoint_name =  model_root / utils.get_checkpoint_name(checkpoint_folder, config)
    training_data = utils.load_training_data(fnames, data_root)
    X, Y = training_data['train']['X'], training_data['train']['Y']

    m =  get_model(config, X, Y)

    if restore:
        if checkpoint_epoch is not None:
            checkpoint_name = str(checkpoint_name)+f'_{checkpoint_epoch}'

        m.load_from_checkpoint(checkpoint_name)

    return m

@ex.predict
def predict(config, m, testing_data):
    console.rule('predicting')

    jitted_pred = m.predict_latents
    def pred_fn(XS): 
        pred_mu, pred_var = jitted_pred(XS)
        return pred_mu[:, 0][None, :], pred_var[:, 0, 0][None, :]

    results = collect_results(
        ex,
        m,
        pred_fn,
        testing_data,
        returns_ci = False
    )

    return results


@ex.automain
def main(config, train_flag=True, restore=False, overwrite=None, return_model = False, quick_predict = False):
    stgp.settings.ng_jitter = 1e-5
    stgp.settings.jitter = 1e-5
    settings.kalman_filter_force_symmetric = True
    settings.verbose = True
    settings.cvi_ng_exploit_space_time = True
    stgp.settings.avoid_s_cholesky = True
    settings.ng_samples = config['ng_samples']

    console.print(config)
    console.print(jax.devices())

    fold = config['fold']
    noise_fold = config['noise_fold']
    fnames = get_data_file_names(fold, noise_fold)

    name = utils.get_unique_experiment_name(config)
    data_root, results_root, checkpoint_folder = utils.get_and_ensure_folder_structure()

    training_data = utils.load_training_data(fnames, data_root)
    testing_data = utils.load_test_data(fnames, data_root)

    # get model
    m = ex.model_function(config, restore=restore)
    console.rule('m before training')


    lc_0 = m.get_objective()
    if train_flag:
        print('Training')
        start = timer()
        trainer = NatGradTrainer(m, enforce_psd_type='laplace_gauss_newton_delta_u')
        maxiters = 1
        lc, _ = trainer.train(1.0, maxiters, callback=progress_bar_callback(maxiters))

        end = timer()
        training_time = end - start
        m.checkpoint(utils.get_checkpoint_name(checkpoint_folder, config))
        lowest_checkpoint_epoch = stgp.trainers.callbacks._lowest_epoch

    else:
        lc = None
        training_time = None
        lowest_checkpoint_epoch = None


    if config['checkpoint']:
        if lowest_checkpoint_epoch is not None:
            console.rule(f'using checkpoint {lowest_checkpoint_epoch}')
            checkpoint_name = str(utils.get_checkpoint_name(checkpoint_folder, config))+f'_{lowest_checkpoint_epoch}'
            m.load_from_checkpoint(checkpoint_name)

    console.rule('m after training')
    m.print()

    if quick_predict:
        testing_data = {'train': training_data['train'], 'test': testing_data['test']}

    testing_data['train'] =  training_data['train']

    if True:
        results = ex.predict_function(config, m, testing_data)

    try:
        ex.log_scalar('training_time', training_time)
    except Exception as e:
        print('problem saving training_time!!') 

    results['meta'] = {
        'lc': lc,
        'training_time': training_time,
        'lowest_epoch': lowest_checkpoint_epoch
    }

    console.rule('metrics')
    console.print(results['metrics'])

    if (overwrite == True) or train_flag:
        console.rule(f"saving to : {results_root/ f'{name}.pickle'}")
        pickle.dump(results, open(results_root/ f'{name}.pickle', "wb" ) )
        ex.add_artifact(results_root/ f'{name}.pickle')

    console.rule('END')


