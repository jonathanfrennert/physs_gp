import scipy.io
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root = Path('data')

ns_mat = scipy.io.loadmat(str(root / 'NS.mat'))

def clean_ns(data):
    # see https://github.com/maziarraissi/PINNs/blob/master/main/continuous_time_identification%20(Navier-Stokes)/NavierStokes.py
    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    
    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    
    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1

    return x, y, t, u, v, p

columns = ['x', 'y', 't', 'u', 'v', 'p']
res = clean_ns(ns_mat)

data = np.squeeze(np.hstack([res_i[:, None] for res_i in res]))
data_df = pd.DataFrame(data, columns=columns)
data_df.to_csv(root / 'NS.csv', index=False)
