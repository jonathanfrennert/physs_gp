import jax
import jax.numpy as np
import numpy as onp
import chex

def _ice_helper(callback, XS, X, s=None, P=1):

    if s is None: raise NotImplementedError()

    # XS must only contain the s 'column'
    chex.assert_rank(XS, 1)

    N, D = X.shape
    NS = XS.shape[0]

    C_idx = list(set(onp.arange(D))-set([s]))

    # batch XS
    # Repeat XS N times into shape  NS x N  
    # each row is just each element of XS repeated N times
    XS_tiled = np.tile(XS, [N, 1]).T
    chex.assert_shape(XS_tiled, [NS, N])

    # Convert to NS - N format
    XS_stacked = np.hstack(XS_tiled)[:, None]
    chex.assert_shape(XS_stacked, [N*NS, 1])

    # sort X
    Xc = X[:, C_idx]
    Xc_tiled = np.tile(Xc, [NS, 1])
    X_concat = np.hstack([XS_stacked, Xc_tiled])

    sort_idx = [s] + C_idx
    reverse_sort_idx = np.array(sort_idx).argsort()

    X_sorted = X_concat[:, reverse_sort_idx]

    return callback(X_sorted)

def ice(pred_fn, XS, X, s=None, P=1):
    """
    Let s be the column we wish to understand the relevance of, and C the complement then we wish to compute
        f_s = E_p(X_C) [ f(X_s, X_c]]  

    In this function we compute the batched version by batching over XS
    """

    N, D = X.shape
    NS = XS.shape[0]

    pred_X, _ = _ice_helper(
        pred_fn, XS, X, s, P
    )


    # ensure consisent shape for multi outputs
    pred_X = np.reshape(pred_X, [P, NS*N])

    return np.transpose(pred_X.T.reshape(NS, N, P), [1, 0, 2])

pdp = ice

def min_c_ice(pred_fn, XS, X, s=None, P=1):
    N, D = X.shape
    NS = XS.shape[0]

    pred_X = ice(pred_fn, XS, X, s, P)

    min_val = pred_X[:, np.argmin(XS), :]

    res = pred_X - min_val[:, None, :]

    return res

def max_c_ice(pred_fn, XS, X, s=None, P=1):
    N, D = X.shape
    NS = XS.shape[0]

    pred_X = ice(pred_fn, XS, X, s, P)

    max_val = pred_X[:, np.argmax(XS), :]

    res = pred_X - max_val[:, None, :]

    return res

def diff_ice(pred_fn, XS, X, s=None, P=1):

    N, D = X.shape
    NS = XS.shape[0]

    diff_fn = jax.vmap(lambda x: jax.jacfwd(pred_fn)(x[None, ...]), 0)

    pred_X, _ = _ice_helper(
        diff_fn, XS, X, s, P
    )

    pred_X = pred_X[..., s]

    return np.transpose(pred_X.reshape(NS, N, P), [1, 0, 2])


