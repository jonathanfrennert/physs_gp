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
from stgp.trainers.standard import VB_NG_ADAM, LBFGS, LikNoiseSplitTrainer, ADAM
from stgp import settings
from stgp.trainers.callbacks import progress_bar_callback, checkpoint_callback_wrapper
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern52, ScaledMatern52
from stgp.kernels.diff_op import SecondOrderDerivativeKernel_2D, SecondOrderDerivativeKernel_1D, SecondOrderDerivativeKernel_1D, SecondOrderSpaceFirstOrderTimeDerivativeKernel_2D, FirstOrderDerivativeKernel, SecondOrderDerivativeKernel, SecondOrderOnlyDerivativeKernel, RemoveDiffDim
from stgp.likelihood import Gaussian, PowerLikelihood, ProductLikelihood
from stgp.models import GP
from stgp.means.mean import SecondOrderDerivativeMean_2D, SecondOrderSpaceFirstOrderTimeDerivativeMean_2D, SecondOrderDerivativeMean, FirstOrderDerivativeMean
from stgp.transforms import LinearTransform, OutputMap, MultiOutput, DataLatentPermutation, Independent
from stgp.core.model_types import get_model_type
from stgp.transforms.pdes import DifferentialOperatorJoint, HeatEquation2D, Pendulum1D, DampedPendulum1D, AllenCahn
from stgp.approximate_posteriors import MeanFieldApproximatePosterior, MeanFieldConjugateGaussian, ConjugateGaussian, FullConjugateGaussian, FullGaussianApproximatePosterior
from stgp.computation.solvers.euler import euler
from stgp.kernels import SpatioTemporalSeperableKernel, Matern32, RBF 
from stgp.transforms.sdes import LTI_SDE, LTI_SDE_Full_State_Obs

from stgp.zoo.diff import diff_vgp

from setup_data import get_data_file_names

from stdata.grids import create_spatial_grid, pad_with_nan_to_make_grid

from experiment_utils import utils

# Create experiment
ex = Experiment(__file__)

# For pretty printing
console = Console()

@ex.configs
def get_config():
    num_epochs = 20000
    #num_epochs = 10
    #num_epochs = 6000

    options = {
        'fold': [0],
        'noise_fold': [0],
        'name': ['ac'],
        'sif': ['~/projects/deep_kernel/deep_kernel_stgp.sif'],
        'lib': ['stgp'],
        Split(): [
            {
                'model': ['autoip'],
                'lik_noise': [0.001],
                'collocation_noise': [0.001],
                'lengthscale': [[0.1, 0.1]],
                'whiten': [True, False],
                'adam_lr': [0.001],
                'ell_samples': [100],
                'num_colocation': [10],
                'kernel': ['rbf'],
                'inference': ['adam'],
                'checkpoint': [False, True] # we need to run twice, with and without checkpoint, to get an accurate estimate of run time
            }
        ],
        'max_iters': [num_epochs],
        'seed': [0],
    }

    return  get_all_permutations(options)

def get_model(config, X, Y):

    if config['model'] == 'autoip':

        if True:
            X_colocation = create_spatial_grid(0.0, 1.0, -1.0, 1.0, config['num_colocation'], config['num_colocation'])

            Y_colocation = np.copy(X_colocation)*0.0
            Y_colocation[:, 0] = np.NaN

            Y_pde = np.hstack([Y, Y*0.0])

            X = np.vstack([X, X_colocation])
            Y = np.vstack([Y_pde, Y_colocation])
        else:
            Y = np.hstack([Y, Y*0.0])

        if config['kernel'] == 'rbf':
            base_kernel = ScaleKernel(RBF(input_dim = 2, lengthscales = config['lengthscale']), 1.0)
        elif config['kernel'] == 'matern':
            base_kernel = ScaleKernel(Matern32(input_dim = 2, lengthscales = config['lengthscale']), 1.0)

        def add_pde_transform(diff_op_prior):
            prior_output_1, prior_output_2 = OutputMap(
                diff_op_prior, 
                [[0], [0, 1, 2]], 
            )

            pde_output = AllenCahn(prior_output_2, train=False)

            prior = MultiOutput([
                prior_output_1,
                pde_output
            ])

            return prior

        diff_kern = SecondOrderSpaceFirstOrderTimeDerivativeKernel_2D(base_kernel)

        Z = None
        print('X:', X.shape, ' Y.shape: ', Y.shape)
        m = diff_vgp(X, Y, lik_var = config['lik_noise'], time_diff = None, space_diff= None, diff_kern = diff_kern, base_kernel = base_kernel, fix_y = True, ell_samples=config['ell_samples'], prior_fn = add_pde_transform, whiten=config['whiten'], Z=Z)

    else:
        raise RuntimeError()

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

    jitted_pred = objax.Jit(m.predict_latents, m.vars())
    def pred_fn(XS): 
        pred_mu, pred_var = jitted_pred(XS)
        return pred_mu[:, 0][None, :], pred_var[:, 0, 0][None, :]

    latent_pred_fn = lambda XS: batch_predict(XS, pred_fn, batch_size=100, verbose=True, axis=1, ci=False)

    results = collect_results(
        ex,
        m,
        latent_pred_fn,
        testing_data,
        returns_ci = False
    )

    return results

@ex.automain
def main(config, train_flag=True, restore=False, overwrite=None, return_model = False, quick_predict = False):
    stgp.settings.ng_jitter = 1e-5
    stgp.settings.jitter = 1e-5

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
    print(m.get_objective())
    m.print()

    if train_flag:
        console.rule('training')
        # the training time includes the jit time 
        start = timer()

        if config['checkpoint']:
            callback = checkpoint_callback_wrapper(
                progress_bar_callback(config['max_iters']),  
                model=m, 
                checkpoint_every=100, 
                checkpoint_name_callback=lambda epoch: utils.get_checkpoint_name(checkpoint_folder, config, epoch=epoch),
                checkpoint_lowest_val = True
            )
        else:
            callback = progress_bar_callback(config['max_iters'])

        if config['inference'] == 'adam':
            # TODO: log training time
            lc, _ = ADAM(m).train(config['adam_lr'], config['max_iters'], callback=callback)
        else:
            trainer = VB_NG_ADAM(m, enforce_psd_type='retraction')
            lc, _ = trainer.train([config['adam_lr'], 0.001], [config['max_iters'], [1, 1]], callback=callback)

        end = timer()
        training_time = end - start
        m.checkpoint(utils.get_checkpoint_name(checkpoint_folder, config))
        lowest_checkpoint_epoch = stgp.trainers.callbacks._lowest_epoch

    else:
        lc = None
        training_time = None
        lowest_checkpoint_epoch = None

    console.rule('m after training')
    m.print()

    if config['checkpoint']:
        if lowest_checkpoint_epoch is not None:
            console.rule(f'using checkpoint {lowest_checkpoint_epoch}')
            checkpoint_name = str(utils.get_checkpoint_name(checkpoint_folder, config))+f'_{lowest_checkpoint_epoch}'
            m.load_from_checkpoint(checkpoint_name)

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
        'lowest_epoch': lowest_checkpoint_epoch
    }

    console.rule('metrics')
    console.print(results['metrics'])

    if (overwrite == True) or train_flag:
        console.rule(f"saving to : {results_root/ f'{name}.pickle'}")
        pickle.dump(results, open(results_root/ f'{name}.pickle', "wb" ) )
        ex.add_artifact(results_root/ f'{name}.pickle')

    console.rule('END')

