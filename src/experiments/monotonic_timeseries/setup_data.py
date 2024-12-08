
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

FOLDS = [100] #DATA SIZE
NUM_FOLDS = len(FOLDS)


def get_data(x, seed=0):
    return 20*np.sin(100*x)+100*x    

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
        x_all = np.linspace(0, 1, 1000)
        X_all = x_all[..., None]
        f_all = get_data(x_all)[:, None]

        x_train = np.linspace(0, 1, FOLDS[i])
        X_train = x_train[..., None]
        f_train = get_data(x_train)[:, None]

        X_vis = X_all
        f_vis = f_all

        # add random noise
        np.random.seed(0)
        sig = 0.001
        Y_train = f_train + sig*np.random.randn(*f_train.shape)
        Y_vis = f_vis + sig*np.random.randn(*f_vis.shape)

    
        if False:
            plt.scatter(X_vis, Y_vis)
            plt.scatter(X_train, Y_train)
            plt.show()
            exit()

        
        # no train-test splits for now
        fnames = get_data_file_names(i)
        print(f'======== {i} ======')
        print(f'X_train: {X_train.shape}')
        print(f'Y_train: {Y_train.shape}, {np.mean(Y_train, axis=0)}, {np.std(Y_train, axis=0)}')
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







