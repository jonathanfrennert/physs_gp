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

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from rich.console import Console
from timeit import default_timer as timer

import stgp
from stgp import settings
from stgp.kernels import RBF, ScaleKernel, BiasKernel, Kernel, Matern32, Matern52, ScaledMatern52, ScaledMatern32, SpatioTemporalSeperableKernel, ScaledMatern72
from stgp.zoo.diff import diff_vgp, diff_hierarchical_vgp, diff_hierarchical_sde_vgp
from stgp.transforms import OutputMap, MultiOutput
from stgp.transforms.pdes import  DampedPendulum1D
from stgp.trainers.callbacks import  progress_bar_callback
from stgp.trainers.standard import VB_NG_ADAM, LBFGS, LikNoiseSplitTrainer, ADAM 

from setup_data import get_data_file_names

from experiment_utils import utils

# Create experiment
ex = Experiment(__file__)

# For pretty printing
console = Console()

@ex.configs
def get_config():
    num_epochs = 1000

    options = {
        'fold': [0],
        'name': ['pendulum'],
        'sif': ['~/projects/deep_kernel/deep_kernel_stgp.sif'],
        'lib': ['stgp'],
        'colocation_lik_var': [0.001],
        Split(): [
            {
                'model': ['hierarchical_sde'],
                'lik_var': [0.01],
                'lengthscale': [1.0],
                'whiten': ['n/a'],
                'inference': ['vi'],
                'M': ['none'],
                'type': ['sde'],
                'kernel': ['scaled-matern72'],
                'ell_samples': [100],
                'num_colocation': [10, 100, 500, 1000],
                'parallel': ['auto'],
            },
            {
                'model': ['vgp'],
                'lik_var': [0.01],
                'lengthscale': [1.0],
                'whiten': [False, True],
                'inference': ['adam'],
                'M': ['none'],
                'type': ['vgp'],
                'kernel': ['rbf'],
                'ell_samples': [100],
                'num_colocation': [10, 100, 500, 1000],
                'parallel': [False],
            },
        ],

        'max_iters': [num_epochs],
        'seed': [0],
    }


    return  get_all_permutations(options)

def get_model(config, X, Y):
    fold = config['fold']
    fnames = get_data_file_names(fold)
    data_root, results_root, checkpoint_folder = utils.get_and_ensure_folder_structure(mkdir =False)
    X_all = utils.load_test_data(fnames, data_root)['vis']['X']

    X, Y = add_colocation(config, X, Y, X_all)

    if config['kernel'] == 'rbf':
        time_kernel = RBF(
            input_dim = 1, lengthscales = [config['lengthscale']], active_dims=[0]
        )
    elif config['kernel'] == 'matern32':
        time_kernel = Matern32(
            input_dim = 1, lengthscales = [config['lengthscale']], active_dims=[0]
        )

    elif config['kernel'] == 'matern52':
        time_kernel = Matern52(
            input_dim = 1, lengthscales = [config['lengthscale']], active_dims=[0]
        )

    elif config['kernel'] == 'scaled-matern52':
        time_kernel = ScaledMatern52(
            input_dim = 1, lengthscales = [config['lengthscale']], active_dims=[0], variance=1.0
        )

    elif config['kernel'] == 'scaled-matern72':
        time_kernel = ScaledMatern72(
            input_dim = 1, lengthscales = [config['lengthscale']], active_dims=[0], variance=1.0
        )


    def add_pde_transform(diff_op_prior):
        prior_output_1, prior_output_2 = OutputMap(
            diff_op_prior, 
            [[0], [0, 1, 2]], 
        )

        prior = MultiOutput([
            prior_output_1,
            DampedPendulum1D(prior_output_2, g=1.0, l=1.0, b=0.2, train=False)
        ])

        return prior

    if config['M'] != 'none':
        breakpoint()

    # add colocation points
    if config['model'] == 'vgp':
        base_kernel = ScaleKernel(time_kernel)

        m = diff_vgp(X, Y, lik_var = [config['lik_var'], config['colocation_lik_var']], time_diff = 2, space_diff= None, base_kernel = base_kernel, fix_y = True, ell_samples=config['ell_samples'], Z=None, whiten=config['whiten'], prior_fn = add_pde_transform)

    elif config['model'] == 'hierarchical_vgp':
        base_kernel = ScaleKernel(time_kernel)

        m = diff_hierarchical_vgp(X, Y, lik_var = config['lik_var'], time_diff = 2, space_diff= None, base_kernel = base_kernel, fix_y = True, ell_samples=config['ell_samples'], Z=None, prior_fn = add_pde_transform)

    elif config['model'] == 'hierarchical_sde':
        # we only care up to the second derivative
        m = diff_hierarchical_sde_vgp(X, Y, time_diff = 2, space_diff = None, time_kernel = time_kernel, space_kernel = None, lik_var = [config['lik_var'], config['colocation_lik_var']], fix_y = True, ell_samples=config['ell_samples'], prior_fn = add_pde_transform, keep_dims=[0, 1, 2], parallel=config['parallel'], verbose=True)

    else:
        raise NotImplementedError()

    m.print()
    print(m.get_objective())
    return m

def add_colocation(config, X_pde, Y_pde, X_all):

    Y_pde = np.hstack([Y_pde, np.zeros_like(Y_pde)])

    # add additional colocation points across the whole input spcae
    X_colocation = np.linspace(np.min(X_all), np.max(X_all), config['num_colocation'])[:, None]
    Y_colocation = np.hstack([X_colocation * np.nan, np.zeros_like(X_colocation)])

    X  = np.vstack([X_pde, X_colocation])
    Y  = np.vstack([Y_pde, Y_colocation])

    return X, Y

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

@ex.automain
def main(config, train_flag=True, restore=False, overwrite=None, return_model = False):
    stgp.settings.ng_jitter = 1e-5
    stgp.settings.jitter = 1e-5
    stgp.settings.verbose = True

    console.rule('Config')
    console.print(config)
    console.rule('Devices')
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
    m.print()

    if train_flag:
        # the training time includes the jit time 
        start = timer()

        print(m.get_objective())
        try:
            console.rule('training')

            if config['inference'] == 'batch':
                # TODO: log training time
                lc, _ = LikNoiseSplitTrainer(m, lambda: LBFGS(m), 0.4).train(None, config['max_iters'], callback=progress_bar_callback(config['max_iters']))

            elif config['inference'] == 'vi':
                lc, _ = LikNoiseSplitTrainer(
                    m,
                    lambda: VB_NG_ADAM(m, enforce_psd_type='laplace_gauss_newton_delta_u'),
                    0.4,
                ).train([0.01, 0.1], [config['max_iters'], [1, 1]], callback=progress_bar_callback(config['max_iters']))


            else:
                lc, _ = LikNoiseSplitTrainer(m, lambda: ADAM(m), 0.4).train(0.01, config['max_iters'], callback=progress_bar_callback(config['max_iters']))


        except RuntimeError as e:
            print('===== AN ERROR OCCURED =====')
            print(e)
            lc = None
            print(m.get_objective())


        end = timer()
        training_time = end - start
        m.checkpoint(utils.get_checkpoint_name(checkpoint_folder, config))

    else:
        lc = None
        training_time = None

    console.rule('m after training')
    m.print()



    if True:
        console.rule('predicting')

        pred_fn = lambda XS: m.predict_latents(XS, squeeze=True)



        if config['type'] == 'gp':
            jitted_pred_fn = objax.Jit(pred_fn, m.vars())
        else :
            jitted_pred_fn = pred_fn

        def pred_fn(XS): 
            pred_mu, pred_var = jitted_pred_fn(XS)

            lik1 = m.likelihood.likelihood_arr[0].likelihood_arr[0].variance

            if len(pred_var.shape) == 1:
                pred_mu, pred_var = pred_mu[None, :], pred_var[None, :]
                
            elif len(pred_var.shape) == 2:
                pred_mu, pred_var =  pred_mu.T, pred_var.T
            elif len(pred_var.shape) == 3:
                pred_mu, pred_var =  pred_mu.T, np.diagonal(pred_var, axis1=1, axis2=2).T
            else:
                pred_mu, pred_var =   pred_mu[:, :, 0].T, np.diagonal(pred_var, axis1=2, axis2=3)[:, 0, :].T

            # add the likelihood noise so we predict y on the first output
            return pred_mu, pred_var+lik1

        latent_pred_fn = lambda XS: batch_predict(XS, pred_fn, batch_size=1000, verbose=True, axis=1, ci=False)

        testing_data['train'] = training_data['train']

        from stgp.computation.gaussian import log_gaussian_scalar
        def metric_callback(ex, YS, pred_mu, pred_var, prefix):
            nlpd = -np.mean(jax.vmap(log_gaussian_scalar, [0, 0, 0])(YS, pred_mu, pred_var))
            ex.log_scalar(f'{prefix}_nlpd', nlpd)
            return {f'{prefix}_nlpd': nlpd}

        results = collect_results(
            ex,
            m,
            latent_pred_fn,
            testing_data,
            returns_ci = False,
            callback = metric_callback
        )

    try:
        ex.log_scalar('training_time', training_time)
    except Exception as e:
        print('problem saving training_time!!')
        

    results['meta'] = {
        'lc': lc,
        'training_time': training_time,
        'lik_var_1': np.array(m.likelihood.likelihood_arr[0].likelihood_arr[0].variance),
        'lik_var_2': np.array(m.likelihood.likelihood_arr[1].likelihood_arr[0].variance),
    }

    console.rule('metrics')
    console.print(results['metrics'])

    if (overwrite == True) or train_flag:
        console.rule(f"saving to : {results_root/ f'{name}.pickle'}")
        pickle.dump(results, open(results_root/ f'{name}.pickle', "wb" ) )
        ex.add_artifact(results_root/ f'{name}.pickle')

    console.rule('END')


