import sys

sys.path = ['../../'] + sys.path
sys.path = ['../'] + sys.path

# Local Run 
sys.path.append('../../') 
# Cluster Run
sys.path.append('../') 

import os

# TRY TO REDUCE MEMORY ERRORS
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=.80
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import objax
import jax
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
jax_config.update('jax_disable_jit', False)

#jax.config.update('jax_check_tracer_leaks', True)

import sdem
from sdem import Experiment
from sdem.utils import read_yaml, get_all_permutations, print_dict, Split
from sdem.modelling.prediction import collect_results

import stdata
from stdata.prediction import batch_predict, st_batch_predict
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
from stgp.trainers.callbacks import progress_bar_callback
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern52, ScaledMatern52
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D, SecondOrderDerivativeKernel_1D, SecondOrderDerivativeKernel_1D, SecondOrderSpaceFirstOrderTimeDerivativeKernel_2D, FirstOrderDerivativeKernel, SecondOrderDerivativeKernel, SecondOrderOnlyDerivativeKernel, ClosedFormRBFFirstOrderDerivativeKernel
from stgp.likelihood import Gaussian, PowerLikelihood, ProductLikelihood
from stgp.models import GP
from stgp.means.mean import SecondOrderDerivativeMean_2D, SecondOrderSpaceFirstOrderTimeDerivativeMean_2D, SecondOrderDerivativeMean, FirstOrderDerivativeMean, SecondOrderOnlyDerivativeMean
from stgp.transforms import LinearTransform, OutputMap, MultiOutput, DataLatentPermutation, Independent
from stgp.core.model_types import get_model_type
from stgp.transforms.pdes import DifferentialOperatorJoint, HeatEquation2D, Pendulum1D, DampedPendulum1D, AllenCahn
from stgp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian, FullGaussianApproximatePosterior, FullConjugatePrecisionGaussian
from stgp.computation.solvers.euler import euler
from stgp.kernels import SpatioTemporalSeperableKernel, Matern32, RBF, ScaledMatern72, ScaledMatern32, ScaledMatern52
from stgp.transforms.sdes import LTI_SDE, LTI_SDE_Full_State_Obs

from stgp.zoo.diff import diff_gp, diff_vgp, diff_hierarchical_sde_vgp, diff_hierarchical_sparse_sde_vgp
from stgp.means.mean import FirstOrderDerivativeMean, SecondOrderDerivativeMean
from stgp.kernels.diff_op import FirstOrderDerivativeKernel, FirstOrderDerivativeKernel_2D, SecondOrderDerivativeKernel, SecondOrderOnlyDerivativeKernel, FirstOrderDerivativeKernel_3D
from stgp.transforms.pdes import DifferentialOperatorJoint, LorenzSystem, LotkaVolterraSystem
from stgp.models import GP
from stgp.zoo.diff import diff_gp
from stgp.kernels import RBF, ScaleKernel, ScaledMatern32, Matern32, ScaledMatern72
from stgp.trainers.standard import LBFGS
from stgp.trainers.callbacks import progress_bar_callback
from stgp.transforms.multi_output import LMC
from stgp.transforms import Independent, OutputMap, MultiOutput
from stgp.likelihood import Gaussian, ProductLikelihood
from stgp.approximate_posteriors import FullGaussianApproximatePosterior, MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, FullConjugateGaussian

from stgp.trainers.standard import VB_NG_ADAM, NatGradTrainer, ADAM
from stgp.trainers.callbacks import progress_bar_callback
from stgp.data import SpatioTemporalData, Data, MultiOutputTemporalData, TemporallyGroupedData
from stgp.transforms.sdes import LTI_SDE_Full_State_Obs, LTI_SDE_Full_State_Obs_With_Mask
from scipy.cluster.vq import kmeans2

from stgp.zoo.phi_ml import helmholtz_3D

from setup_data import get_data_file_names

from experiment_utils import utils

import matplotlib.pyplot as plt


# Create experiment
ex = Experiment(__file__)

# For pretty printing
console = Console()

def fix_prediction_shapes(fn):
    def _wrapper(XS):
        mu, var = fn(XS)
        return np.squeeze(mu).T, np.squeeze(var).T

    return _wrapper

@ex.configs
def get_config():
    num_epochs = 10000
    #num_epochs = 1000

    options = {
        'fold': [0],
        'noise_fold': [0],
        'name': ['laser'],
        'sif': ['~/projects/deep_kernel/deep_kernel_stgp.sif'],
        'lib': ['stgp'],
        'ls': [1.0],
        'lik_var': [0.01],
        Split(): [
            {
                'model': ['pigp'],
                'M': [100],
                'hierarchical': [True],
                'adam_lr': [1e-2],
                'enforce_psd_type': ['laplace_gauss_newton_delta_u'],
                'ng_lr': [0.1],
                'meanfield': [False],
                'lik_fix_percent': [0.4],
                'type': ['cvi'],
                'data_type': ['all'],
                'parallel': ['auto'],
                'temporally_grouped': [True],
                'minibatch': [10],
                Split(): [
                    {
                        'pretrain_ng': [False],
                        'pretrain_max_iter': ['none']
                    }
                ]
            }
        ],
        'jitter': [1e-5],
        'ng_jitter': [1e-5],
        'max_iters': [num_epochs],
        'ng_samples': [100],
        'seed': [0],
    }

    return  get_all_permutations(options)


def get_model(config, training_data, XS):
    if config['data_type'] == 'all':
        X, Y = training_data['train']['X'], training_data['train']['Y']
        print('DATA: ', X.shape, Y.shape)
    else:
        raise RuntimeError()

    ls = config['ls']

    base_time_kernels = [
        ScaledMatern32(input_dim=1, lengthscales=[24.0], variance=1, active_dims=[0])
        for i in range(2)
    ]

    base_space_kernels = [
        RBF(input_dim=1, lengthscales=[ls], active_dims=[1]) * RBF(input_dim=1, lengthscales=[ls], active_dims=[2])
        for i in range(2)
    ]

    space_diff_kernel = None

    if config['model'] == 'gp':

        m = helmholtz_3D(
            X, 
            Y,
            time_kernel = base_time_kernels,
            space_kernel = base_space_kernels,
            lik_var = config['lik_var'],
            fix_y = True,
            verbose=True,
            model='batch_gp'
        )


    elif config['model'] == 'pigp':

        temporally_grouped = config['temporally_grouped']
        parallel = config['parallel']

        if parallel == 'auto':
            if jax.devices()[0].device_kind == 'cpu':
                console.log('running locally -- sequential filter used')
                parallel = False
            else:
                console.log('running externally -- parallel filter used')
                parallel = True

        minibatch_size = config['minibatch']

        if minibatch_size == 'none':
            minibatch_size = None

        st_data = TemporallyGroupedData(X, Y)

        # construct inducing points on the spatial locations
        X_space = X[:, 1:]
        Z_s = kmeans2(X_space, config['M'], minit="points")[0]

        m = helmholtz_3D(
            X, 
            Y,
            time_kernel = base_time_kernels,
            space_kernel = base_space_kernels,
            space_diff_kernel = space_diff_kernel,
            hierarchical = config['hierarchical'],
            Zs = Z_s,
            lik_var = config['lik_var'],
            fix_y = True,
            meanfield = config['meanfield'],
            verbose=True,
            parallel=parallel,
            temporally_grouped = temporally_grouped,
            minibatch_size=minibatch_size
        )
    else:
        raise RuntimeError()

    return m


@ex.model
def model(config, model_root: Path = '.', restore=True, checkpoint_epoch=None):
    model_root = Path(model_root)
    fold = config['fold']
    fnames = get_data_file_names(fold)

    # dont make folder structure if we are restoring as it should already exist
    data_root, results_root, checkpoint_folder = utils.get_and_ensure_folder_structure(mkdir = ~ restore)

    checkpoint_name =  model_root / utils.get_checkpoint_name(checkpoint_folder, config)
    training_data = utils.load_training_data(fnames, data_root)

    testing_data = utils.load_test_data(fnames, data_root)
    XS = testing_data['test']['X']

    m =  get_model(config, training_data, XS)


    if restore:
        if checkpoint_epoch is not None:
            checkpoint_name = str(checkpoint_name)+f'_{checkpoint_epoch}'

        m.load_from_checkpoint(checkpoint_name)

    print('INIT OBJ: ', m.get_objective())

    return m


@ex.automain
def main(config, train_flag=True, restore=False, overwrite=None, return_model = False, quick_predict = False, quick_train=False, predict = True, checkpoint_epoch = None, suffix = None):
    if True:
        import os
        print('PIPD: ', os.getppid())

    np.random.seed(config['seed'])

    stgp.settings.ng_jitter = config['jitter']
    stgp.settings.jitter = config['jitter']
    stgp.settings.ng_samples = config['ng_samples']
    settings.kalman_filter_force_symmetric = True
    settings.verbose = True
    settings.cvi_ng_batch = False
    settings.cvi_ng_exploit_space_time = True

    console.print(config)
    console.print(jax.devices())

    fold = config['fold']
    fnames = get_data_file_names(fold)

    name = utils.get_unique_experiment_name(config)
    data_root, results_root, checkpoint_folder = utils.get_and_ensure_folder_structure()

    training_data = utils.load_training_data(fnames, data_root)
    testing_data = utils.load_test_data(fnames, data_root)

    # get model
    m = ex.model_function(config, restore=restore, checkpoint_epoch=checkpoint_epoch)
    X, Y = np.array(m.data.X), np.array(m.data.Y)

    console.rule('m before training')

    base_checkpoint_name = str(utils.get_checkpoint_name(checkpoint_folder, config))

    if restore:
        breakpoint()

    #print(m.get_objective())

    console.rule('training')
    m.print()

    if train_flag:
        try:
            m.likelihood.fix()
            m.print()

            # the training time includes the jit time 
            start = timer()


            if config['type'] == 'cvi':
                enforce_psd_type = config['enforce_psd_type']
                if enforce_psd_type == 'none':
                    enforce_psd_type = None
                trainer = LikNoiseSplitTrainer(m, lambda: VB_NG_ADAM(m, ng_nan_max_attempt=10, enforce_psd_type=enforce_psd_type), config['lik_fix_percent'])

                if config['pretrain_ng']:
                    lc_ng, _ = trainer.trainer_with_lik_held.ng_trainer.train(config['ng_lr'], config['pretrain_max_iter'], callback=progress_bar_callback(config['pretrain_max_iter']))

                max_iter = config['max_iters']
                lc, _ = trainer.train([config['adam_lr'], config['ng_lr']], [max_iter, [1, 1]], callback=progress_bar_callback(max_iter), raise_error = False)

                if config['pretrain_ng']:
                    lc = [lc_ng, lc]

            else:
                trainer = LikNoiseSplitTrainer(m, lambda: ADAM(m), config['lik_fix_percent'])
                max_iter = config['max_iters']
                lc, _ = trainer.train(config['adam_lr'], max_iter, callback=progress_bar_callback(max_iter), raise_error = False)

            end = timer()
            training_time = end - start

            m.checkpoint(base_checkpoint_name)
        except Exception as e:
            # it is likely that a nan was encounted
            console.rule('finishing early! probably due to nans')
            print(e)
            lc = None
            training_time = None
    else:
        lc = None
        training_time = None
        if restore:
            m.load_from_checkpoint(base_checkpoint_name)


    if quick_train:
        if config['type'] == 'cvi':
            trainer = NatGradTrainer(m, enforce_psd_type='laplace_gauss_newton_delta_u')
            #trainer = NatGradTrainer(m)
            trainer.train(0.1, 1)
            print(m.get_objective())



    console.rule('m after training')
    print(m.get_objective())
    m.print()

    testing_data['train'] =  training_data['train']

    if quick_predict:
        testing_data = {}
        testing_data['train'] = training_data['train']



    if not predict:
        # presave in case predicitng fails
        results = {}
        results['meta'] = {
            'lc': lc,
        }

        if (overwrite == True) or train_flag:
            console.rule(f"saving to : {results_root/ f'{name}.pickle'}")
            pickle.dump(results, open(results_root/ f'{name}.pickle', "wb" ) )
            ex.add_artifact(results_root/ f'{name}.pickle')
        exit()

    if True:
        console.rule('predicting')

        jitted_pred_output = lambda XS: m.predict_f(XS)

        def pred_fn_output(XS): 
            print(XS.shape)
            pred_mu, pred_var = jitted_pred_output(XS)
            pred_mu = np.squeeze(pred_mu)
            pred_var = np.squeeze(pred_var)
            return pred_mu.T, pred_var.T

        if config['type'] == 'cvi':
            batched_latent_pred_fn = lambda XS: st_batch_predict(None, XS, pred_fn_output, batch_size=1000, verbose=True, out_dim=2, transpose_pred=True)


        else:
            batched_latent_pred_fn = lambda XS: batch_predict(XS, pred_fn_output, batch_size=1000, verbose=True, axis=1, ci=False)
        

        results = collect_results( ex, m, fix_prediction_shapes(batched_latent_pred_fn), testing_data, returns_ci = False)


    results['meta'] = {
        'lc': lc,
        'checkpoint_epoch': checkpoint_epoch,
        'training_time': training_time
    }


    try:
        ex.log_scalar('training_time', training_time)
    except Exception as e:
        print('problem saving training_time!!')

    console.rule('metrics')
    console.print(results['metrics'])

    if (overwrite == True) or train_flag:
        if suffix is not None:
            name = name + f'_{suffix}'
        console.rule(f"saving to : {results_root/ f'{name}.pickle'}")
        pickle.dump(results, open(results_root/ f'{name}.pickle', "wb" ) )
        ex.add_artifact(results_root/ f'{name}.pickle')

    console.rule('END')


