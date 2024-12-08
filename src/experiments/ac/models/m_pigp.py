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
from stgp.likelihood import Gaussian, PowerLikelihood, ProductLikelihood
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
                'model': ['pigp'],
                'lik_noise': [0.001],
                'collocation_noise': [0.001],
                'lengthscale': [[0.1, 0.1]],
                'whiten': [False],
                'adam_lr': [0.001],
                'ng_lr': [[0.1, 0.1]],
                'ell_samples': [100],
                'M': [10, 15, 20],
                Split(): [
                    {
                        'train_Z': [True],
                        'ng_schedule_iters': [10000],
                    },
                    {
                        'train_Z': [False],
                        'ng_schedule_iters': [None],
                    },
                ],
                'num_colocation': [[20, 10], [10, 10]],
                'parallel': ['auto'],
                'train_type': ['adam'],
                'hierarchical': [False, True],
                'checkpoint': [False], # we need to run twice, with and without checkpoint, to get an accurate estimate of run time,
                'pretrain_ng': [True],
                'max_iters': [num_epochs-100],
                'schedule': ['constant', 'log'],
                Split(): [

                    {
                        'experimental_time_weight': [True],
                        'experimental_time_weight_type': ['simple']
                    },
                    {
                        'experimental_time_weight': [False],
                        'experimental_time_weight_type': ['n/a']
                    },
                ],
                Split():[
                    {
                        'ng_momentum': [False],
                        'ng_momentum_rate': ['n/a'],
                    },
                ],
                Split(): [
                    {   
                        'delta_method': [False],
                        'ng_samples': [10],
                        'ng_f_samples': [10]
                    }
                ]
            }
        ],
        'seed': [0],
    }
    return  get_all_permutations(options)

def get_model(config, X, Y):

    # create colocation points
    if True:
        X_colocation = create_spatial_grid(0.0, 1.0, -1.0, 1.0, config['num_colocation'][0], config['num_colocation'][1])

        Y_colocation = np.copy(X_colocation)*0.0
        Y_colocation[:, 0] = np.NaN

        Y_pde = np.hstack([Y, Y*0.0])

        X = np.vstack([X, X_colocation])
        Y = np.vstack([Y_pde, Y_colocation])
    else:
        Y = np.hstack([Y, Y*0.0])

    time_kernel = ScaledMatern72(
        input_dim = 1, lengthscales = [config['lengthscale'][0]], active_dims=[0], variance=1.0
    )

    space_kernel = RBF( lengthscales=[config['lengthscale'][1]], active_dims=[1], input_dim=1)

    Z_s = np.linspace(-1, 1, config['M'])[:, None]

    def add_pde_transform(diff_op_prior):
        prior_output_1, prior_output_2 = OutputMap(
            diff_op_prior, 
            [[0], [0, 2, 1]], 
        )

        pde_output = AllenCahn(prior_output_2, train=False)

        prior = MultiOutput([
            prior_output_1,
            pde_output
        ])

        return prior


    if config['model'] == 'pigp':

        m = diff_cvi_sde_vgp(
            X, 
            Y, 
            num_latents = 1,
            time_diff = 1, 
            space_diff = -2, 
            time_kernel = time_kernel, 
            space_kernel = space_kernel, 
            lik_var = [config['lik_noise'], config['collocation_noise']],
            fix_y = True, 
            Zs = Z_s,
            ell_samples=config['ell_samples'], 
            prior_fn = add_pde_transform,
            keep_dims = [0, 1], #only applies to the temporal diffs
            temporally_grouped=True,
            train_Z = False,
            multioutput_prior=True,
            minibatch_size=None,
            parallel=config['parallel'],
            verbose=True,
            hierarchical=config['hierarchical']
        )


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

    if config['experimental_time_weight_type'] == 'cumsum':
        t = np.array(m.data.X[:, 0])
        _, t_int = np.unique(np.squeeze(t), return_inverse =True)
        stgp.settings.experimental_precomputed_segements = t_int
        stgp.settings.experimental_precomputed_num_segements = np.unique(stgp.settings.experimental_precomputed_segements).shape[0]
        

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
    if True:
        stgp.settings.ng_jitter = 1e-5
        stgp.settings.jitter = 1e-5
    else:
        stgp.settings.ng_jitter = 1e-7
        stgp.settings.jitter = 1e-7

    settings.kalman_filter_force_symmetric = True
    settings.verbose = True
    settings.cvi_ng_exploit_space_time = True
    settings.ng_samples = config['ng_samples']

    if config['experimental_time_weight']:
        if config['experimental_time_weight_type'] == 'simple':
            settings.experimental_simple_time_weight = True
            settings.experimental_cumsum_time_weight = False
        elif config['experimental_time_weight_type'] == 'cumsum':
            settings.experimental_simple_time_weight = False
            settings.experimental_cumsum_time_weight = True
        else:
            raise RuntimeError('not found')


    if config['ng_f_samples'] != 'none':
        settings.ng_f_samples = config['ng_f_samples']

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
        # the training time includes the jit time 
        start = timer()

        console.rule('training')
        if config['ng_f_samples'] == 'none':
            if config['delta_method']:
                enforce_type = 'laplace_gauss_newton_delta_u'
            else:
                enforce_type = 'laplace_gauss_newton'
        else:
            if config['delta_method']:
                enforce_type = 'laplace_gauss_newton_delta_u_mc_f'
            else:
                enforce_type = 'laplace_gauss_newton_mc_f'

        try:
            if config['train_type'] == 'adam':
                trainer = VB_NG_ADAM(m, enforce_psd_type=enforce_type, ng_schedule=config['schedule'])

                if config['pretrain_ng']:
                    # use a smaller lengthscale for pretraining
                    trainer.ng_trainer.train(
                            [0.01, 0.01], 
                            100, 
                            callback=progress_bar_callback(100),
                            momentum=config['ng_momentum'], 
                            momentum_rate=config['ng_momentum_rate']
                        )

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


                if config['ng_schedule_iters'] is None:
                    ng_schedule_iters = 0
                    lc_1 = []
                else:
                    ng_schedule_iters = config['ng_schedule_iters']
                    print(f"training for {config['ng_schedule_iters']}")
                    lc_1, _ = trainer.train(
                        [config['adam_lr'], config['ng_lr']], 
                        [config['ng_schedule_iters'], [1, 1]], 
                        callback=progress_bar_callback(config['ng_schedule_iters']), 
                        ng_momentum=config['ng_momentum'], 
                        ng_momentum_rate=config['ng_momentum_rate']
                    )

                if config['train_Z']:
                    # train_Z
                    m.prior.parent[0].parent.get_sparsity_list()[0].raw_Z.release()
                    trainer = VB_NG_ADAM(m, enforce_psd_type=enforce_type, ng_schedule=config['schedule'])
                    print('now training Z')
                    m.print()

                print(f"training for {config['max_iters']-ng_schedule_iters}")
                lc_2, _ = trainer.train(
                    [config['adam_lr'], [config['ng_lr'][1], config['ng_lr'][1]]], 
                    [config['max_iters']-ng_schedule_iters, [1, 1]], 
                    callback=progress_bar_callback(config['max_iters']-ng_schedule_iters), 
                    ng_momentum=config['ng_momentum'], 
                    ng_momentum_rate=config['ng_momentum_rate']
                )
                lc = [lc_1, lc_2]

            elif config['train_type'] == 'ng':
                trainer = NatGradTrainer(m, enforce_psd_type=enforce_type)
                lc, _ = trainer.train(config['ng_lr'], config['max_iters'], callback=progress_bar_callback(config['max_iters']))
        except RuntimeError as e:
            print(e)
            lc = None

        lc_1 = m.get_objective()
        print(lc_0-lc_1)

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

