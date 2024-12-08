import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from stdata.utils import save_to_pickle
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

SEED = 0
NOISE_ADDED = [0.01]

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

    # Load data
    data_df = pd.read_csv(data_root / 'pendulum_dt_0.03_g_1.0_l_1.0_b_0.2_n_1000.csv')

    x_all = np.array(data_df['x'])
    y_all = np.array(data_df['y'])

    for i in range(len(NOISE_ADDED)):
        # split training into train/test splits
        fnames = get_data_file_names(i)

        np.random.seed(SEED)
        #use first 200 as training data and add random gaussian noise
        N = 200
        x_train = x_all[:N]
        y_train = y_all[:N] + NOISE_ADDED[i]*np.random.randn(N)

        # Use rest of data for testing
        x_test = x_all[N:]
        y_test = y_all[N:] + NOISE_ADDED[i]*np.random.randn(x_test.shape[0])

        if True:
            plt.scatter(x_train, y_train)
            plt.scatter(x_test, y_test)
            plt.show()

        # only use 20 points for training
        idx = np.random.choice(np.arange(N), 20)
        x_train = x_train[idx]
        y_train = y_train[idx]


        #  training data
        X_train = x_train[:, None]
        Y_train = y_train[:, None]

        X_test = x_test[:, None]
        Y_test = y_test[:, None]

        X_vis = x_all[:, None]
        Y_vis = y_all[:, None]

        print(f'======== {i} ======')
        print(f'X_train: {X_train.shape}')
        print(f'Y_train: {Y_train.shape}')
        print(f'X_test: {X_test.shape}')
        print(f'Y_test: {Y_test.shape}')
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
                'Y': Y_vis
            }
        }

        save_to_pickle(train_data, data_path / fnames['train'])
        save_to_pickle(test_data, data_path / fnames['test'])
        save_to_pickle(raw_data, data_path / fnames['raw'])

if __name__ == '__main__':
    experiment_root = Path('.')
    data_root = Path('../../data/pendulum/data')

    setup(data_root, experiment_root)





