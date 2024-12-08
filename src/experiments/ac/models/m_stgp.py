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

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from rich.console import Console
from timeit import default_timer as timer

import stgp
from stgp.zoo.multi_output import gp_regression, lmc_regression, lmc_drd_regression, gprn_regression, gprn_drd_regression, gprn_drd_nv_regression
from stgp.trainers import ScipyTrainer, GradDescentTrainer, NatGradTrainer
from stgp.trainers.callbacks import  progress_bar_callback
from stgp.trainers.standard import VB_NG_ADAM, LBFGS, LikNoiseSplitTrainer
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern52, ScaledMatern52

from setup_data import get_data_file_names

from experiment_utils import utils

# Create experiment
ex = Experiment(__file__)

# For pretty printing
console = Console()

@ex.configs
def get_config():
    num_epochs = 20000

    options = {
        'fold': [0],
        'noise_fold': [0],
        'name': ['ac'],
        'sif': ['~/projects/deep_kernel/deep_kernel_stgp.sif'],
        'lib': ['stgp'],
        Split(): [
            {
                'model': ['gp'],
                'lik_noise': [0.01],
                'lengthscale': [[0.1, 0.1]],
                'whiten': ['n/a']
            }
        ],
        'max_iters': [num_epochs],
        'seed': [0],
    }


    return  get_all_permutations(options)

def get_model(config, X, Y):



    if config['model'] == 'gp':
        kernels = [
            RBF(input_dim=2, lengthscales=config['lengthscale']) 
        ]
        m = gp_regression(X, Y, kernels=kernels, lik_noise=config['lik_noise'], inference='Batch')

    else:
        raise RuntimeError()

    return m



@ex.model
def model(config, model_root: Path = '.', restore=True):
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
        m.load_from_checkpoint(checkpoint_name)

    return m

@ex.automain
def main(config, train_flag=True, restore=False, overwrite=None, return_model = False):
    stgp.ng_jitter = 1e-5
    stgp.jitter = 1e-5

    console.print(config)
    console.print(jax.devices())

    fold = config['fold']
    noise_fold = config['noise_fold']
    fnames = get_data_file_names(fold, noise_fold)

    name = utils.get_unique_experiment_name(config)
    data_root, results_root, checkpoint_folder = utils.get_and_ensure_folder_structure()

    training_data = utils.load_training_data(fnames, data_root)
    testing_data = utils.load_test_data(fnames, data_root)

    testing_data['train'] = training_data['train']

    # get model
    m = ex.model_function(config, restore=restore)
    console.rule('m before training')
    m.print()

    if True:
        console.rule('training')

        # TODO: log training time
        lc, _ = LBFGS(m).train(None, config['max_iters'], callback=progress_bar_callback(config['max_iters']))

        m.checkpoint(utils.get_checkpoint_name(checkpoint_folder, config))

    else:
        lc = None

    console.rule('m after training')
    m.print()

    if True:
        console.rule('predicting')

        def pred_fn(XS): 
            pred_mu, pred_var = m.predict_f(XS, squeeze=False)
            # ensure rank 2
            return pred_mu[:, 0, :].T, pred_var[:, :, 0, 0].T

        latent_pred_fn = lambda XS: batch_predict(XS, pred_fn, batch_size=1000, verbose=True, axis=1, ci=False)

        results = collect_results(
            ex,
            m,
            latent_pred_fn,
            testing_data,
            returns_ci = False
        )
        

    results['meta'] = {
        'lc': lc,
    }

    console.rule('metrics')
    console.print(results['metrics'])

    if (overwrite == True) or train_flag:
        console.rule(f"saving to : {results_root/ f'{name}.pickle'}")
        pickle.dump(results, open(results_root/ f'{name}.pickle', "wb" ) )
        ex.add_artifact(results_root/ f'{name}.pickle')

    console.rule('END')

