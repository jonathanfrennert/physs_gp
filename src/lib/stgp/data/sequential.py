""" Numpy operations for dealing sequential data """
import numpy as onp
import jax.numpy as np
from jax import jit
from functools import partial
import chex
from scipy.cluster.vq import kmeans2

def pad_with_nan_to_make_grid(X, Y):
    """
    Adds additional missing obersations to X and Y to ensure that they are defined on a grid.

    In:
        X: N x D
        Y: N x P
    Out:
        X: N_grid x D
        Y: N_grid x P
    """

    process_y = Y is not None
    # Ensure correct format
    chex.assert_rank(X, 2)

    if process_y:
        chex.assert_rank(Y, 2)
        chex.assert_equal(X.shape[0], Y.shape[0])
        P = Y.shape[1]

    N = X.shape[0]

    #construct target grid
    unique_time = onp.unique(X[:, 0])
    unique_space = onp.unique(X[:, 1:], axis=0)

    Nt = unique_time.shape[0]
    Ns = unique_space.shape[0]

    X_tmp = onp.tile(onp.expand_dims(unique_space, 0), [Nt, 1, 1])

    time_tmp = onp.tile(unique_time, [Ns]).reshape([Nt, Ns], order='F')

    X_tmp = X_tmp.reshape([Nt*Ns, -1])

    time_tmp = time_tmp.reshape([Nt*Ns, 1])

    #X_tmp is the full grid
    X_tmp = onp.hstack([time_tmp, X_tmp])

    #Find the indexes in X_tmp that we need to add to X to make a full grid
    _X = onp.vstack([X,  X_tmp])

    if process_y:
        _Y = onp.nan*onp.zeros([_X.shape[0], P])

    _, idx = onp.unique(_X, return_index=True, axis=0)
    idx = idx[idx>=N]

    X_to_add = _X[idx, :]
    X_grid = onp.vstack([X, X_to_add])

    if process_y:
        Y_to_add = _Y[idx, :]
        Y_grid = onp.vstack([Y, Y_to_add])
    else:
        Y_grid = None


    return X_to_add.shape[0], X_grid, Y_grid

def order_sequentially_np(X, Y = None):
    """
    lexsort uses the final column as the primary sort key and then sorts by each column from the last
        1 -  roll the columns so the time is the last column
        2 - get idx of new ordering
        3 - roll X so that time is the first axis again

    NOTE: assumes that X can be represented as a spatio-temporal grid

    In:
        X: N x D
        (optional) Y: N x P

    Out:
        X: Nt x Ns x D
        (optional) Y: Nt x Nd x P
    """
    _X = np.copy(X)
    chex.assert_rank(X, 2)

    if Y is not None:
        chex.assert_equal(X.shape[0], Y.shape[0])
        chex.assert_rank(Y, 2)
        P = Y.shape[1]

    # X is on a space time grid so can choose any time index to find out how many
    #  spatial points there are
    time_zero = X[0, 0]

    # Get unique rows (time, space, features) to remove duplicates and sorts
    _, unique_idx, reverse_idx = onp.unique(X, axis=0, return_index = True, return_inverse=True)

    # Get index to sort by time points and then spatial points
    #   we need the index so that we can undo the sort later
    # required so that all spatial points are consistenly organised

    # Get unique points
    X = X[unique_idx]

    # Since X is a spatio-temporal grid we can just extract the spatial points at the first time
    X_spatial = X[X[:, 0]==time_zero][:,1:]

    # Put time axis as last axis os that this is sorted first
    #  it does not matter the order that the spatial dimensions get sorted
    X = onp.roll(X, -1, axis=1)

    grid_size = X_spatial.shape[0]
    time_points = int(X.shape[0]/grid_size)

    if False:
        #TODO: not needed as unique does this for us
        # Sort in space and time
        idx = onp.lexsort(X.T)
    else:
        idx = np.arange(X.shape[0])


    X = X[idx]

    if Y is not None:
        Y = Y[unique_idx][idx]

    #reset time axis
    X = onp.roll(X, 1, axis=1)

    #reshape for grid structure
    X = onp.reshape(X, [time_points, grid_size, X.shape[1]])

    if Y is not None:
        Y = onp.reshape(Y, [time_points, grid_size, P])

        return unique_idx, reverse_idx, idx, X, Y

    return unique_idx, reverse_idx, idx, X, None

def add_temporal_points(XS: 'Data', X: 'Data'):
    """
    Only add new temporal points to X from XS. Whilst keeping the spatial points in X.
    """

    Nt = XS.Nt
    Ns = X.Ns

    # Get spatial locations in X
    spatial_locations = X.X_space

    # Unique time points in XS
    unique_time_points = XS.X_time

    # ensure rank 1
    unique_time_points = onp.reshape(unique_time_points, [-1])

    # Create spatio-temporal grid from unique_time_points and spatial_locations
    new_st_points = onp.hstack([
        np.repeat(unique_time_points, X.Ns)[:, None],
        np.tile(spatial_locations, [Nt,  1])
    ])

    return new_st_points

def get_minimal_time_groups(X, Y=None, verbose=True):
    """
    Groups X, Y by time and padds each group so there is the same number of points.

    First column of X must be time
    """

    # TODO: this is just a hack to support not passing Y
    if Y is None:
        no_Y = True
        Y = np.ones([X.shape[0], 1])
    else:
        no_Y = False

    # Get unique rows (time, space, features) to remove duplicates and sorts
    _, unique_idx, reverse_idx = onp.unique(X, axis=0, return_index = True, return_inverse=True)
    
    # Get unique points
    X = X[unique_idx]
    Y = Y[unique_idx]
    
    Nd = X.shape[1]
    P = Y.shape[1]
    
    # combine so that indexing is consisent
    
    D = onp.hstack([X, Y])
    
    # get all spatial points
    X_all_st = X[:, 1:]
    
    # group by time
    # as D is already sorted in time, this will not change the order
    D_groups = onp.split(D, onp.unique(D[:, 0], return_index=True)[1][1:])
    
    # run k means -- these will be used as filler points
    max_spatial_points = max([D_t.shape[0] for D_t in D_groups])
    Z_s = kmeans2(X_all_st, max_spatial_points, minit="points")[0]
    Ns = max_spatial_points
    
    if verbose:
        print(f'max number of spatial points: {max_spatial_points}')
    
    _X = []
    _Y = []

    actual_data_idx = []
    
    for i, D_t in enumerate(D_groups):
        D_ns = D_t.shape[0]
        # add dummy values
        X_to_add = Z_s[:(Ns-D_ns)]
        Y_to_add = onp.zeros([X_to_add.shape[0], P])*onp.nan

        # add time 
        X_to_add = onp.hstack([
            onp.tile(onp.array([[D_t[0, 0]]]), X_to_add.shape[0]).T,
            X_to_add
        ])
        
        X_group = D_t[:, :Nd]
        Y_group = D_t[:, Nd:]

        X_group_padded = onp.vstack([X_group, X_to_add])
        Y_group_padded = onp.vstack([Y_group, Y_to_add])

        _X.append(X_group_padded)
        _Y.append(Y_group_padded)

        N_total = X_group_padded.shape[0]

        actual_data_idx.append(np.arange(N_total - X_to_add.shape[0]) + N_total * i)

    actual_data_idx = np.hstack(actual_data_idx)

    if no_Y:
        return unique_idx, reverse_idx, actual_data_idx, onp.array(_X), None
    else:
        return unique_idx, reverse_idx, actual_data_idx, onp.array(_X), onp.array(_Y)
        
