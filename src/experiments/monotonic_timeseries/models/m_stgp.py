import sys
# Local Run 
sys.path.append('../../') 
# Cluster Run
sys.path.append('../') 

import objax
import jax
from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

import sdem
from sdem import Experiment
from sdem.utils import read_yaml, get_all_permutations, print_dict, Split
from sdem.modelling.prediction import collect_results

import stdata
from stdata.prediction import batch_predict
from stdata.grids import create_spatial_grid, pad_with_nan_to_make_grid



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
from stgp.trainers.callbacks import progress_bar_callback, checkpoint_callback_wrapper
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern52, ScaledMatern52, ScaledMatern72
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D, SecondOrderDerivativeKernel_1D, SecondOrderDerivativeKernel_1D, SecondOrderSpaceFirstOrderTimeDerivativeKernel_2D, FirstOrderDerivativeKernel, SecondOrderDerivativeKernel, SecondOrderOnlyDerivativeKernel
from stgp.likelihood import Gaussian, PowerLikelihood, ProductLikelihood, Probit
from stgp.models import GP
from stgp.means.mean import SecondOrderDerivativeMean_2D, SecondOrderSpaceFirstOrderTimeDerivativeMean_2D, SecondOrderDerivativeMean, FirstOrderDerivativeMean, SecondOrderOnlyDerivativeMean
from stgp.transforms import LinearTransform, OutputMap, MultiOutput, DataLatentPermutation, Independent
from stgp.core.model_types import get_model_type
from stgp.transforms.pdes import DifferentialOperatorJoint, HeatEquation2D, Pendulum1D, DampedPendulum1D, AllenCahn
from stgp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian, FullGaussianApproximatePosterior
from stgp.computation.solvers.euler import euler
from stgp.kernels import SpatioTemporalSeperableKernel, Matern32, RBF, ScaledMatern72, ScaledMatern32
from stgp.transforms.sdes import LTI_SDE, LTI_SDE_Full_State_Obs

from stgp.zoo.diff import diff_gp, diff_vgp, diff_hierarchical_sde_vgp, diff_hierarchical_sparse_sde_vgp
from stgp.zoo.sde_diff import diff_cvi_sde_vgp
from stgp.zoo.gps import batch_gp

from setup_data import get_data_file_names

from experiment_utils import utils

# Create experiment
ex = Experiment(__file__)

# For pretty printing
console = Console()

@ex.configs
def get_config():
    options = {
        'fold': [0],
        'name': ['monotonic_timeseries'],
        'sif': ['~/projects/deep_kernel/deep_kernel_stgp.sif'],
        'lib': ['stgp'],
        Split(): [
            {
                'model': ['pigp'],
                'n_colocation': [300],
                'parallel': ['auto'],
                'lengthscale': [0.1],
                'ng_delta_method': [False],
                'trainer': ['ng_adam'],
                'adam_lr': [0.01],
                'ng_lr': [0.1]
            },
            {
                'model': ['vgp'],
                'n_colocation': [300],
                'parallel': ['n/a'],
                'ng_delta_method': ['n/a'],
                'trainer': ['adam'],
                'adam_lr': [0.01],
                'ng_lr': ['n/a'],
                'lengthscale': [0.1],
            },
            {
                'model': ['gp'],
                'n_colocation': ['n/a'],
                'parallel': ['n/a'],
                'ng_delta_method': ['n/a'],
                'trainer': ['adam'],
                'adam_lr': [0.01],
                'ng_lr': ['n/a'],
                'lengthscale': [0.1],
            },
        ],
        'max_iters': [10000],
        #'max_iters': [200],
        'seed': [0],
    }

    return  get_all_permutations(options)

def add_colocation_points(config, X, Y):
    Y = np.hstack([Y, np.ones_like(Y)])
    N_colocation = config['n_colocation']
    X_colocation = np.linspace(np.min(X), np.max(X), N_colocation)[:, None]
    Y_colocation = np.ones([N_colocation, Y.shape[1]])
    Y_colocation[:, 0] = np.NaN

    X = np.vstack([X, X_colocation])
    Y = np.vstack([Y, Y_colocation])

    return X, Y

def get_model(config, X, Y):
    lik_noise = 1.0

    if config['model'] == 'pigp':
        X, Y = add_colocation_points(config, X, Y)

        time_kernel = ScaledMatern32(
            input_dim = 1, lengthscales = [config['lengthscale']], active_dims=[0], variance=1.0
        )

        m = diff_cvi_sde_vgp(
            X=X,
            Y=Y,
            num_latents=1,
            time_diff=1,
            time_kernel = time_kernel,
            space_diff=None,
            ell_samples=100,
            parallel=config['parallel'],
            #lik_arr=[Gaussian(lik_noise), Probit(nu=1e-1)]
            lik_arr=[Gaussian(lik_noise), Probit(nu=1.0)]
        )
    elif config['model'] == 'vgp':
        X, Y = add_colocation_points(config, X, Y)

        time_kernel = ScaledMatern32(
            input_dim = 1, lengthscales = [config['lengthscale']], active_dims=[0], variance=1.0
        )

        m = diff_vgp(
            X=X,
            Y=Y,
            time_diff=1,
            space_diff=None,
            base_kernel = time_kernel,
            ell_samples=100,
            lik_arr=[Gaussian(lik_noise), Probit(nu=1e-1)],
            whiten=True
        )

    elif config['model'] == 'gp':
        time_kernel = ScaledMatern32(
            input_dim = 1, lengthscales = [config['lengthscale']], active_dims=[0], variance=1.0
        )
        m = batch_gp(
            X = X, 
            Y= Y,
            kernel = ScaledMatern32(
                input_dim = 1, lengthscales = [config['lengthscale']], active_dims=[0], variance=1.0
            ),
            likelihood = Gaussian(lik_noise)
        )

    return m


@ex.model
def model(config, model_root: Path = '.', restore=True):
    model_root = Path(model_root)
    fold = config['fold']
    fnames = get_data_file_names(fold)

    # dont make folder structure if we are restoring as it should already exist
    data_root, results_root, checkpoint_folder = utils.get_and_ensure_folder_structure(mkdir = ~ restore)

    checkpoint_name =  model_root / utils.get_checkpoint_name(checkpoint_folder, config)
    training_data = utils.load_training_data(fnames, data_root)
    X, Y = training_data['train']['X'], training_data['train']['Y']

    m =  get_model(config, X, Y)

    if restore:
        m.load_from_checkpoint(checkpoint_name)

    return m

@ex.predict
def predict(config, m, testing_data):
    console.rule('predicting')

    jitted_pred = m.predict_y
    def pred_fn(XS): 
        pred_mu, pred_var = jitted_pred(XS)
        return pred_mu.T, pred_var.T

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

    settings.verbose = True
    settings.cvi_ng_exploit_space_time = False # we can go even fast we implement this

    console.print(config)
    console.print(jax.devices())

    fold = config['fold']
    fnames = get_data_file_names(fold)

    name = utils.get_unique_experiment_name(config)
    data_root, results_root, checkpoint_folder = utils.get_and_ensure_folder_structure()

    training_data = utils.load_training_data(fnames, data_root)
    testing_data = utils.load_test_data(fnames, data_root)

    # get model
    m = ex.model_function(config, restore=restore)


    console.rule('m before training')
    print(m.get_objective())

    m.print()

    if train_flag:
        console.rule('training')
        # the training time includes the jit time 
        start = timer()

        callback = progress_bar_callback(config['max_iters'])
        if config['trainer'] == 'adam':
            lc, _ = ADAM(m).train(config['adam_lr'], config['max_iters'], callback=callback)
        elif config['trainer'] == 'ng_adam':
            if config['ng_delta_method']:
                enforce_type = 'gauss_newton_delta_u'
            else:
                enforce_type = 'gauss_newton'

            if True:
                trainer = VB_NG_ADAM(m, enforce_psd_type=enforce_type)
                pretrain_iter = 1000
                #lc, _ = trainer.ng_trainer.train(config['ng_lr'], pretrain_iter, callback=progress_bar_callback(pretrain_iter))
                
                lc, _ = trainer.train([config['adam_lr'], config['ng_lr']], [config['max_iters'], [1, 1]], callback=callback)
            else:
                pretrain_iter = config['max_iters']
                trainer = NatGradTrainer(m, enforce_psd_type=enforce_type)
                #lc, _ = trainer.train(config['ng_lr'][0], pretrain_iter, callback=progress_bar_callback(pretrain_iter))
                lc, _ = trainer.train(config['ng_lr'], pretrain_iter, callback=progress_bar_callback(pretrain_iter))
        else:
            raise NotImplementedError()

        end = timer()
        training_time = end - start
        m.checkpoint(utils.get_checkpoint_name(checkpoint_folder, config))

    else:
        lc = None
        training_time = None

    console.rule('m after training')
    m.print()

    testing_data['train'] = training_data['train']

    if quick_predict:
        testing_data = {'test': testing_data['test'], 'train': training_data['train']}

    if True:
        results = ex.predict_function(config, m, testing_data)
    try:
        ex.log_scalar('training_time', training_time)
    except Exception as e:
        print('problem saving training_time!!') 

    results['meta'] = {
        'lc': lc,
        'training_time': training_time,
    }

    console.rule('metrics')
    console.print(results['metrics'])

    if (overwrite == True) or train_flag:
        console.rule(f"saving to : {results_root/ f'{name}.pickle'}")
        pickle.dump(results, open(results_root/ f'{name}.pickle', "wb" ) )
        ex.add_artifact(results_root/ f'{name}.pickle')

    console.rule('END')

