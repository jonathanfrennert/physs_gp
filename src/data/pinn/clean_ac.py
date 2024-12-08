import scipy.io
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root = Path('data')

ac_mat = scipy.io.loadmat(str(root / 'AC.mat'))

T = ac_mat['tt'] # 1 x 201
X = ac_mat['x'] # 1 x 512
Y = ac_mat['uu'] # 512 x 201

# 512 x 201
T_grid, X_grid = np.meshgrid(T[0], X[0])

T_flat = np.reshape(T_grid, [-1, 1])
X_grid = np.reshape(X_grid, [-1, 1])
Y_flat = np.reshape(Y, [-1, 1])

data = np.hstack([T_flat, X_grid, Y_flat])

data_df = pd.DataFrame(data, columns=['t', 'x', 'y'])

if False:
    plt.scatter(data_df['t'], data_df['x'], c=data_df['y'])
    plt.show()

data_df.to_csv(root / 'AC.csv', index=False)
