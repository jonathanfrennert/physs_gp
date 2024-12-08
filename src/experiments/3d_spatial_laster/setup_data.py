import numpy as np
import pickle
from pathlib import Path

import sys
sys.path.append('../../experiment_utils/')
sys.path.append('../experiment_utils/')
sys.path.append('experiment_utils/')
from utils import get_spatial_binned_data
from stdata.grids import create_spatial_grid, create_geopandas_spatial_grid

import stdata
import geopandas as gpd

from stdata.model_selection import normalise_df
from stdata.utils import save_to_pickle
import matplotlib.pyplot as plt
from stgp.data import SpatioTemporalData

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import numpy as np
from sklearn.model_selection import train_test_split
from stdata.utils import datetime_to_epoch
from stdata.model_selection import normalise_df


NUM_FOLDS = 5

def get_data_file_names(fold):
    return {
        'raw': f'raw_data_{fold}.pickle', 
        'train': f'train_data_{fold}.pickle',
        'test': f'test_data_{fold}.pickle', 
    }

def make_spatial_grid(data_df, n_x=None, n_y=None, x_col=None, y_col=None, padding_min=0, padding_max = 0):
    min_x = np.min(data_df[x_col])-padding_min
    max_x = np.max(data_df[x_col])+padding_max
    min_y = np.min(data_df[y_col])-padding_min
    max_y = np.max(data_df[y_col])+padding_max
    
    size_x = (max_x-min_x)/n_x
    size_y = (max_y-min_y)/n_y
    
    grid_gdf = create_geopandas_spatial_grid(
        min_x,
        max_x,
        min_y,
        max_y,
        size_x, 
        size_y
    )

    #grid created on x-y format
    N_x = len(np.arange(min_x, max_x-size_x, size_x))
    N_y = len(np.arange(min_y, max_y-size_y, size_y))

    x_locs = np.reshape(np.tile(np.arange(N_x)[:, None].T, (N_y, 1)).T, [-1, 1])
    y_locs = np.tile(np.arange(N_y), [N_x])[:, None]
    
    grid_gdf['x_loc'] = x_locs
    grid_gdf['y_loc'] = y_locs
    
    grid_gdf['grid_id'] = grid_gdf.index
    
    return grid_gdf



def load_data(datasets_root, month, month2=None):
    if month2 is None:
        month2 = month+1

    # Load data
    raw_data_df = pd.read_csv(datasets_root / 'laser.csv')

    # replace id with an int
    raw_data_df['id'] = raw_data_df.groupby('id').ngroup()

    # book keeping
    raw_data_df['datetime'] = pd.to_datetime(raw_data_df['datetime'])
    raw_data_df['gid'] = raw_data_df.index


    data_df = raw_data_df.copy()
    #data_df = select_region(raw_data_df)
    data_df['date'] = pd.to_datetime(data_df['date'] )

    # round datetime to nearest 15 mins
    # data has been smoothed from every 5 mins to every 15 but there are some small artificats in the datatime
    data_df['datetime'] = data_df['datetime'].dt.round('15min')  

    data_df['hour'] = data_df['datetime'].dt.hour

    # only select numeric columns
    data_df = data_df[
        ['id', 'datetime', 'date', 'hour', 'lat', 'lon', 'position_error', 'u', 'v', 'velocity_error', 'gid']
    ]
    
    data_df = data_df[(data_df['date']>=f'2016-{month}-01') & (data_df['date']<f'2016-{month2}-01')]
    
    return data_df

def add_epoch(data_df, dt_column = 'datetime'):
    data_df['epoch'] = datetime_to_epoch(data_df[dt_column])

    return data_df

def normalise_epoch(df, wrt_to_df):
    # convert to hours
    return (df - np.min(wrt_to_df)) / (60 * 60)

def normalise(df, wrt_to_df):
    # will be [epoch, x, y]
    # epoch -> hourly
    # x, y get z normalised
    s1 = normalise_epoch(df[:, [0]], wrt_to_df=wrt_to_df[:, [0]])
    s2 = normalise_df(df[:, 1:], wrt_to=wrt_to_df[:, 1:])
    return np.hstack([s1, s2])

def daily_avg(data_df):
    # compute a daily average of each lat-lon locations -- ie collapse time
    return data_df.copy().groupby(['id', 'date', 'lat', 'lon']).mean().reset_index()

def hourly_avg(data_df):
    return data_df.copy().groupby(['id', 'date', 'hour', 'lat', 'lon']).mean().reset_index()

def lat_lon_to_meters(data_df):
    data_gdf = gpd.GeoDataFrame(
        data_df,
        geometry = gpd.points_from_xy(
            data_df['lon'],
            data_df['lat']
        )
    )
    data_gdf = data_gdf.set_crs('EPSG:4326') 
    # in meters (TODO this is a global coordinate system, might be a more accurate local one?)
    data_gdf = data_gdf.to_crs('EPSG:3857')
    data_gdf['x'] = data_gdf['geometry'].x
    data_gdf['y'] = data_gdf['geometry'].y
    return data_gdf

def get_X(df):
    return np.array(df[['epoch', 'x', 'y']])

def get_Y(df):
    return np.array(df[['u', 'v']])

def select_region(data_df):
    return data_df[
        (data_df['lon'] >= -90) &
        (data_df['lon'] <= -84.5) &
        (data_df['lat'] >= 26) &
        (data_df['lat'] <= 30) 
    ]

def select_region_small(data_df):
    return data_df[
        (data_df['lon'] >= -86.8) &
        (data_df['lon'] <= -85.5) &
        (data_df['lat'] >= 26.3) &
        (data_df['lat'] <= 27.5) 
    ]

def setup(datasets_root, experiment_root):
    results_path = experiment_root / 'results'
    data_path = experiment_root / 'data'

    # Ensure results and data file exists
    results_path.mkdir(exist_ok=True)
    data_path.mkdir(exist_ok=True)

    # Load data
    data_df = load_data(datasets_root, month=2, month2=4)
    data_df = select_region(data_df)
    print(data_df['datetime'].min(), data_df['datetime'].max())
    data_df = data_df[(data_df['datetime'] >= '2016-02-25') & (data_df['datetime'] <= '2016-02-26')]
    print(data_df['datetime'].min(), data_df['datetime'].max())
    data_df = hourly_avg(data_df)

    data_df['datetime'] = data_df['date'] + pd.to_timedelta(data_df['hour'], unit='h')
    data_df = add_epoch(data_df)
    data_df = lat_lon_to_meters(data_df)
    print(data_df['datetime'].min(), data_df['datetime'].max())
    breakpoint()

    #normalise(get_X(data_df), get_X(data_df))
    grid_gdf = make_spatial_grid(data_df, 50, 50, 'lon', 'lat', padding_max=0.01)

    #TODO: we are not guarenting that the test indexes are disjoint
    for fold in range(NUM_FOLDS):
        data_fold_df = data_df.copy()

        # train test 
        train_df, test_df = train_test_split(data_fold_df, random_state=fold, shuffle=True, test_size=0.1)

        X_all, Y_all = get_X(data_fold_df), get_Y(data_fold_df)

        # ===== TRAINING DATA =====

        # raw data
        X_train,  Y_train =  get_X(train_df), get_Y(train_df)


        # ===== TEST DATA =====
        # test of the raw data
        X_test, Y_test = get_X(test_df), get_Y(test_df)

        # ===== NORMALIZE INPUTS ====
        X_train_norm = normalise(X_train, X_train)
        X_test_norm = normalise(X_test, X_train)
        X_all_norm = normalise(X_all, X_train)

        # ===== VIS DATA ====
        X_grid = create_spatial_grid(
            np.min(train_df['x']), 
            np.max(train_df['x']), 
            np.min(train_df['y']), 
            np.max(train_df['y']), 
            100, 
            100
        )

        all_epochs = data_fold_df['epoch'].unique()        

        # use floor so we include first and last epoch
        vis_plots_epochs_index = list(
            np.arange(
                0, 
                len(all_epochs), 
                int(np.floor(len(all_epochs)/5))
            )
        )
        vis_plots_epochs = all_epochs[vis_plots_epochs_index]

        X_vis_arr = []
        X_vis_norm_arr = []
        Y_vis_arr = []

        for vis_epoch in vis_plots_epochs:
            X_vis = np.hstack([
                np.ones([X_grid.shape[0], 1])*vis_epoch,
                X_grid
            ])
            X_vis_norm = normalise(X_vis, X_train)
            Y_vis = None

            X_vis_arr.append(X_vis)
            X_vis_norm_arr.append(X_vis_norm)
            Y_vis_arr.append(Y_vis)

        i = fold
        fnames = get_data_file_names(i)

        print(f'======== {i} ======')
        print(f'X_train: {X_train.shape}')
        print(f'X_train_norm: {X_train_norm.shape}')
        print(f'Y_train: {Y_train.shape}')
        print(f'X_test: {X_test.shape}')
        print(f'X_test_norm: {X_test_norm.shape}')
        print(f'Y_test: {Y_test.shape}')
        print(f'X_all: {X_all.shape}')
        print(f'X_all_norm: {X_all_norm.shape}')
        print(f'Y_all: {Y_all.shape}')
        for X_vis in X_vis_arr:
            print(f'X_vis: {X_vis.shape}')
            print(f'X_vis_norm: {X_vis_norm.shape}')
        print(f'==================')

        train_data = {
            'train': {
                'X': X_train_norm,
                'Y': Y_train
            }
        }

        test_data = {
            'test': {
                'X': X_test_norm,
                'Y': Y_test
            }, 

            'all': {
                'X': X_all_norm,
                'Y': Y_all
            }
        }

        for i in range(len(vis_plots_epochs)):
            test_data[f'vis_{i}'] = {
                'X': X_vis_norm_arr[i],
                'Y': Y_vis_arr[i]
            }

        # Save unnormalised X for plotting
        raw_data = { 
            'df': data_fold_df,
            'vis_plots_epochs': vis_plots_epochs,
            'X_vis': X_vis_arr
        }

        print(f"saving to {data_path / fnames['train']}")
        print(f"saving to {data_path / fnames['test']}")
        print(f"saving to {data_path / fnames['raw']}")
        save_to_pickle(train_data, data_path / fnames['train'])
        save_to_pickle(test_data, data_path / fnames['test'])
        save_to_pickle(raw_data, data_path / fnames['raw'])

if __name__ == '__main__':
    experiment_root = Path('.')
    data_root = Path('../../data/laser/data')

    setup(data_root, experiment_root)
