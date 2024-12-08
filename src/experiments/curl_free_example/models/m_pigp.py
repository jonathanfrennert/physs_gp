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
from stdata.prediction import batch_predict, st_batch_predict
from stdata.grids import create_spatial_grid, pad_with_nan_to_make_grid

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from rich.console import Console
from timeit import default_timer as timer

from setup_data import get_data_file_names
from experiment_utils import utils

import stgp
from stgp.means.mean import FirstOrderDerivativeMean, SecondOrderDerivativeMean
from stgp.kernels.diff_op import FirstOrderDerivativeKernel, FirstOrderDerivativeKernel_2D, SecondOrderDerivativeKernel, SecondOrderOnlyDerivativeKernel, FirstOrderDerivativeKernel_3D
from stgp.transforms.pdes import DifferentialOperatorJoint
from stgp.models import GP
from stgp.zoo.diff import diff_gp
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Linear, ScaledMatern72, ScaledMatern32
from stgp.trainers.standard import LBFGS, ADAM, VB_NG_ADAM, LikNoiseSplitTrainer
from stgp.trainers.callbacks import progress_bar_callback
from stgp.transforms.multi_output import LMC
from stgp.transforms import Independent
from stgp.likelihood import Gaussian

from stgp.trainers import NatGradTrainer

from scipy.cluster.vq import kmeans2

from stgp.zoo.phi_ml import magnetic_field_strength_H

from setup_data import get_data_file_names

from experiment_utils import utils

# Create experiment
ex = Experiment(__file__)

# For pretty printing
console = Console()


@ex.configs
def get_config():
    num_epochs = 5000
    lik_noise = 1e-3

    options = {
        'fold': [0, 1, 2],
        'name': ['curl_free_example'],
        'sif': ['~/projects/deep_kernel/deep_kernel_stgp.sif'],
        'lib': ['stgp'],
        'lik_noise': [0.01],
        'lengthscale': [0.1],
        'whiten': ['n/a'],
        'adam_lr': [0.01],
        'kernel': ['scaled-matern32'],
        'ell_samples': ['n/a'],
        'num_colocation': ['n/a'],
        Split(): [
            {
                'model': ['diff_gp'],
                'M': ['none'],
                Split(): [
                    {
                        'inference': ['batch'],
                        'ng_lr': ['n/a'],
                    },
                    #{
                    #    'inference': ['vi'],
                    #    'ng_lr': [1.0],
                    #}   
                ],
                'hierarchical': [False],
                'parallel': [False],

            },
            {
                'model': ['sde_cvi'],
                Split(): [
                    {
                        'inference': ['vi'],
                        'ng_lr': [1.0],
                        'hierarchical': [False, True],
                        'M': ['none', 5],
                        'parallel': [True],
                        #'parallel': [False],
                    }
                ],
            },
        ],
        'jitter': [1e-7],
        'max_iters': [num_epochs],
        'seed': [0],
    }


    return  get_all_permutations(options)

def add_nan_potential(Y):
    Y = np.array(Y)
    Y = np.hstack([
        np.nan*Y[:, [0]], 
        Y[:, [0]], 
        Y[:, [1]], 
        Y[:, [2]]
    ])
    return Y

def get_model(config, training_data, XS):
    X, Y = np.array(training_data['train']['X']), np.array(training_data['train']['Y'])

    # we want to see the potential function
    Y = add_nan_potential(Y)

    base_kernel_time = ScaledMatern32(input_dim=1, lengthscales=[config['lengthscale']], variance=0.1, active_dims=[0])

    base_kernel_space = ScaleKernel(
        RBF(input_dim=1, lengthscales=[config['lengthscale']], active_dims=[1]) * 
        RBF(input_dim=1, lengthscales=[config['lengthscale']], active_dims=[2])
    ) 

    if config['model'] == 'diff_gp':
        if config['inference'] == 'batch':
            m = magnetic_field_strength_H(
                X=X,
                Y=Y,
                time_kernel = base_kernel_time,
                space_kernel = base_kernel_space,
                lik_var = config['lik_noise'],
                model='batch_gp',
                include_potential_function = True,
            )
        elif config['inference'] == 'vi':
            m = magnetic_field_strength_H(
                X=X,
                Y=Y,
                time_kernel = base_kernel_time,
                space_kernel = base_kernel_space,
                lik_var = config['lik_noise'],
                model='vgp',
                include_potential_function = True,
            )
    if config['model'] == 'sde_cvi':
        # always need some inducing points
        Xs = stgp.data.SpatioTemporalData(X=X, Y=Y, sort=True).X_space

        if config['M'] == 'none':
            Zs = Xs
        else:
            Zs =  kmeans2(Xs, config['M'], minit="points")[0]

        m = magnetic_field_strength_H(
            X=X,
            Y=Y,
            time_kernel = base_kernel_time,
            space_kernel = base_kernel_space,
            lik_var = config['lik_noise'],
            model='sde_cvi',
            include_potential_function = True,
            hierarchical=config['hierarchical'],
            parallel=config['parallel'],
            Zs = Zs
        )
  
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

    return m


@ex.automain
def main(config, train_flag=True, restore=False, overwrite=None, return_model = False, quick_predict = False, quick_train=False, predict = True, checkpoint_epoch = None, suffix = None):
    if True:
        import os
        print('PIPD: ', os.getppid())

    np.random.seed(config['seed'])

    stgp.settings.ng_jitter = config['jitter']
    stgp.settings.jitter = config['jitter']

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

    print('init obj: ', m.get_objective())

    console.rule('training')

    if train_flag:
        try:
            m.likelihood.fix()
            m.print()

            # the training time includes the jit time 
            start = timer()

            if config['inference'] == 'vi':
                trainer = LikNoiseSplitTrainer(m, lambda: VB_NG_ADAM(m), 0.4)
                max_iter = config['max_iters']
                lc, _ = trainer.train([config['adam_lr'], config['ng_lr']], [max_iter, [1, 1]], callback=progress_bar_callback(max_iter), raise_error = False)

            else:
                trainer = LikNoiseSplitTrainer(m, lambda: ADAM(m), 0.4)
                max_iter = config['max_iters']
                lc, _ = trainer.train(config['adam_lr'], max_iter, callback=progress_bar_callback(max_iter), raise_error = False)

            end = timer()
            training_time = end - start

            m.checkpoint(base_checkpoint_name)
        except Exception as e:
            # it is likely that a nan was encounted
            console.rule('finishing early! probably due to nans')
            print(e)
            raise e
            lc = None
            training_time = None
    else:
        lc = None
        training_time = None
        if restore:
            m.load_from_checkpoint(base_checkpoint_name)

    if quick_train:
        if config['inference'] == 'vi':
            trainer = NatGradTrainer(m)
            trainer.train(1.0, 1)



    console.rule('m after training')
    print(m.get_objective())
    m.print()

    testing_data['train'] =  training_data['train']

    if quick_predict:
        testing_data = {}
        testing_data['train'] = training_data['train']

    # add missing potentials
    for key in testing_data.keys():
        if 'Y' in testing_data[key].keys():
            testing_data[key]['Y'] = add_nan_potential(testing_data[key]['Y'])

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
            pred_mu, pred_var = jitted_pred_output(XS)
            pred_mu = np.squeeze(pred_mu)
            pred_var = np.squeeze(pred_var)
            return pred_mu.T, pred_var.T



        results = collect_results( ex, m, pred_fn_output, testing_data, returns_ci = False)


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


