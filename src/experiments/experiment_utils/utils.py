""" Common functions that are usually required for runnign experiments. """
import objax
import jax
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import stdata
from stdata.plots import grid_to_matrix
from stdata.grids import create_spatial_grid, pad_with_nan_to_make_grid,  create_geopandas_spatial_grid
import geopandas as gpd

def get_and_ensure_folder_structure(mkdir=True):
    data_root = Path('../data/')
    results_root = Path('../results/')
    if mkdir:
        results_root.mkdir(exist_ok=True)

    checkpoint_folder = Path('checkpoints')
    if mkdir:
        checkpoint_folder.mkdir(exist_ok=True)

    return data_root, results_root, checkpoint_folder
def get_unique_experiment_name(config: dict) -> str:
    return '{name}_{_id}'.format(name=config['name'], _id=config['experiment_id'])

def load_training_data(fnames, data_root: Path) -> dict:
    return pickle.load(open(data_root / fnames['train'], "rb" ))

def load_test_data(fnames, data_root: Path) -> dict:
    return pickle.load(open(data_root / fnames['test'], "rb" ))

def load_raw_data(fnames, data_root: Path) -> dict:
    return pickle.load(open(data_root / fnames['raw'], "rb" ))

def get_checkpoint_name(checkpoint_folder, config, epoch=None):
    # Unique experiment name
    name = get_unique_experiment_name(config)
    if epoch is not None:
        checkpoint_name = checkpoint_folder / (name+'_'+str(epoch))
    else:
        checkpoint_name = checkpoint_folder / name

    return checkpoint_name

def get_spatial_binned_data(df, grid_gdf = None, padding_min=0.1, padding_max = 0.1, n_x = 10, n_y = 10, x_col='x', y_col='y', return_grid_details=False, return_grid_gdf = True):
    """ 
    Constructs a spatial grid and computes the interesection with df

    Args:
        padding_max: Optional[list, int]: if list corresponding to padding in [x, y]
    """
    data_df = df.copy()
    data_gdf = gpd.GeoDataFrame(
        data_df,
        geometry = gpd.points_from_xy(data_df[x_col], data_df[y_col])
    )

    if type(padding_min) is not list:
        padding_min = [padding_min, padding_min]

    if type(padding_max) is not list:
        padding_max = [padding_max, padding_max]
    
    min_x = np.min(data_df[x_col])-padding_min[0]
    max_x = np.max(data_df[x_col])+padding_max[0]
    min_y = np.min(data_df[y_col])-padding_min[1]
    max_y = np.max(data_df[y_col])+padding_max[1]
    
    size_x = (max_x-min_x)/n_x
    size_y = (max_y-min_y)/n_y

    
    if grid_gdf is None:
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
    data_gdf['data_id'] = data_gdf.index
    
    # compute spatial intersection of data and grid to find out which polygons each data point belond to
    intersection_gdf = gpd.overlay(data_gdf, grid_gdf, how='intersection')

    # get the average value of each polygon
    aggr_inter_gdf = intersection_gdf.groupby(['grid_id', 'x_loc', 'y_loc']).mean(numeric_only=True).reset_index()
    
    data_grid_gdf = pd.merge(
        grid_gdf,
        aggr_inter_gdf,
        how='left',
        on=['grid_id','x_loc', 'y_loc']
    )
    
    if return_grid_details:
        return [
            min_x,
            max_x,
            min_y,
            max_y,
            size_x, 
            size_y
        ], data_grid_gdf

    if return_grid_gdf:
        return grid_gdf, data_grid_gdf

    return data_grid_gdf

