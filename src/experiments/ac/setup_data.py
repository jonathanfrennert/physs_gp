"""
Construct train-testing data on the AC dataset. 

We construct 5 splits by varying the seed used to generate the train-test data and additionally 
    construct 3 versions for each fold by varying the amount of noise added.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from stdata.grids import create_spatial_grid
from stdata.utils import save_to_pickle
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

NUM_FOLDS = 5
SEED = 0
NOISE_ADDED = [0.01, 0.1, 0.5]

def get_data_file_names(fold, noise):
    return {
        'raw': f'raw_data_{fold}_{noise}.pickle', 
        'train': f'train_data_{fold}_{noise}.pickle',
        'test': f'test_data_{fold}_{noise}.pickle', 
    }

def setup(datasets_root, experiment_root):
    results_path = experiment_root / 'results'
    data_path = experiment_root / 'data'

    # Ensure results and data file exists
    results_path.mkdir(exist_ok=True)
    data_path.mkdir(exist_ok=True)

    # Load data
    data_df = pd.read_csv(data_root / 'AC.csv')

    X_all = np.array(data_df[['t', 'x']])
    y_all = np.array(data_df['y'])

    X_vis = create_spatial_grid(0, 1, -1, 1, 100, 100)
    Y_vis = None

    for j in range(NUM_FOLDS):
        for i in range(len(NOISE_ADDED)):
            print(f'Constructing new dataset with fold {j} and noise {NOISE_ADDED[i]}')
            # split training into train/test splits
            fnames = get_data_file_names(j, i)

            # construct random seed by adding the fold number to the seed
            np.random.seed(SEED+j)

            # Set up training data
            # Following AUTOIP paper
            time_idx = X_all[:, 0] < 0.28
            X_train = X_all[time_idx, ...]
            y_train = y_all[time_idx, ...] 

            idx = np.random.choice(np.arange(X_train.shape[0]), 256)
            #idx = np.random.choice(np.arange(X_train.shape[0]), 100)
            X_train = X_train[idx]
            y_train = y_train[idx] + NOISE_ADDED[i]*np.random.randn(X_train.shape[0])
            Y_train = y_train[:, None]

             # Setup testing data
            test_idx = np.random.choice(np.arange(X_all.shape[0]), 1000)

            X_test = X_all[test_idx, ...]
            y_test = y_all[test_idx, ...] + NOISE_ADDED[i]*np.random.randn(X_test.shape[0])
            Y_test = y_test[:, None]

            # Set up vis data
            if False:
                X_vis = X_all
                Y_vis = y_all[:, None]

            print(f'======== {i} ======')
            print(f'X_train: {X_train.shape}')
            print(f'Y_train: {Y_train.shape}')
            print(f'X_test: {X_test.shape}')
            print(f'Y_test: {Y_test.shape}')
            print(f'X_vis: {X_vis.shape}')
            #print(f'Y_vis: {Y_vis.shape}')
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

            # Save unnormalised X for plotting
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
                    'Y': Y_vis,
                    'df': data_df
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
    data_root = Path('../../data/pinn/data')

    setup(data_root, experiment_root)





