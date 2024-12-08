
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from stdata.grids import create_spatial_grid
from stdata.utils import save_to_pickle
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

FOLDS = [5, 10, 20, 50] # spatial size
TIME_POINTS = 50
NUM_FOLDS = len(FOLDS)

def dipole(m, r, r0):
    # H field of a magnetic field
    m = jnp.array(m)
    r = jnp.array(r)
    r0 = jnp.array(r0)
    
    r = r - r0

    norm_R = jnp.sqrt(jnp.sum(jnp.square(r)))
    norm_R_5 = norm_R**5
    norm_R_3 = norm_R**3

    val = (3/norm_R_5)* jnp.dot(m, r) * r - m/norm_R_3
    return (1/(4*jnp.pi)) * val

def get_data(X, seed=0):
    X = jnp.array(X)

    locations = [
        #[-2, -2, 0],
        [0, 0, 0],
        #[2, -2, 0],
        #[1, 0, 0],
    ]
    Y = None

    for loc in locations:
        Y_loc = jax.vmap(
            dipole, 
            [None, 0, None]
        )(
            [0, 1, 0], 
            X, 
            loc
        )

        if Y is None:
            Y = Y_loc
        else:
            Y = Y + Y_loc

    return Y

def get_data_file_names(fold):
    return {
        'raw': f'raw_data_{fold}.pickle', 
        'train': f'train_data_{fold}.pickle',
        'test': f'test_data_{fold}.pickle', 
    }

def setup(datasets_root, experiment_root):
    results_path = experiment_root / 'results'
    data_path = experiment_root / 'data'

    # Ensure results and data file exists
    results_path.mkdir(exist_ok=True)
    data_path.mkdir(exist_ok=True)

    for i in range(NUM_FOLDS):
        # Load data
        NC = 30
        test_range = 3
        train_range = 3
        X_all = create_spatial_grid(-test_range, test_range, -test_range, test_range, NC, NC)
        X_all = np.hstack([
            X_all,
            np.ones([X_all.shape[0], 1])
        ])
        f_all = get_data(X_all)

        X_train = create_spatial_grid(-train_range, train_range, -train_range, train_range, TIME_POINTS, FOLDS[i])
        X_train = np.hstack([
            X_train,
            np.ones([X_train.shape[0], 1])
        ])
        f_train = get_data(X_train)

        X_test = X_all
        f_test = f_all

        X_vis = X_all
        f_vis = f_all

        # add random noise
        np.random.seed(0)
        sig = 0.001
        Y_train = f_train + sig*np.random.randn(*f_train.shape)
        Y_test = f_test + sig*np.random.randn(*f_test.shape)
        Y_vis = f_vis + sig*np.random.randn(*f_vis.shape)

        if False:
            Bx = Y_test[:, 0]
            By = Y_test[:, 1]
            Bx_grid = np.reshape(Bx, [NC, NC])
            By_grid = np.reshape(By, [NC, NC])

            plt.streamplot( X_grid, Y_grid, Bx_grid, By_grid)
            plt.show()

        if True:
            plt.quiver(X_test[:, 0], X_test[:, 1], Y_test[:,0], Y_test[:, 1])
            plt.quiver(X_test[:, 0], X_test[:, 1], f_test[:,0], f_test[:, 1])
            plt.quiver(X_train[:, 0], X_train[:, 1], Y_train[:,0], Y_train[:, 1], color='red')
            plt.show()
            
        breakpoint()

        
        # no train-test splits for now
        fnames = get_data_file_names(i)
        print(f'======== {i} ======')
        print(f'X_train: {X_train.shape}')
        print(f'Y_train: {Y_train.shape}, {np.mean(Y_train, axis=0)}, {np.std(Y_train, axis=0)}')
        print(f'X_test: {X_test.shape}')
        print(f'Y_test: {Y_test.shape}, {np.mean(Y_test, axis=0)}, {np.std(Y_test, axis=0)}')
        print(f'X_vis: {X_vis.shape}')
        print(f'Y_vis: {Y_vis.shape}')
        print(f'==================')

        train_data = {
            'train': {
                'X': X_train,
                'Y': Y_train,
            }
        }

        test_data = {
            'test': {
                'X': X_test,
                'Y': Y_test
            }, 
            'vis': {
                'X': X_vis,
                'Y': Y_vis
            }
        }

        raw_data = {
            'train': {
                'X': X_train,
                'Y': Y_train,
            },
            'test': {
                'X': X_test,
                'Y': Y_test
            },
            'vis': {
                'X': X_vis,
                'Y': Y_vis
            }
        }

        print(f"saving to {data_path / fnames['train']}")
        print(f"saving to {data_path / fnames['test']}")
        print(f"saving to {data_path / fnames['raw']}")
        save_to_pickle(train_data, data_path / fnames['train'])
        save_to_pickle(test_data, data_path / fnames['test'])
        save_to_pickle(raw_data, data_path / fnames['raw'])

if __name__ == '__main__':
    experiment_root = Path('.')
    data_root = Path('.')

    setup(data_root, experiment_root)






