
import jax
from jax import jacfwd, jit
import jax.numpy as np
from jax.lax import scan, associative_scan

from ... import settings 
from ..matrix_ops import cholesky, cholesky_solve, add_jitter
from ..gaussian import log_gaussian, log_gaussian_with_mask
from ...utils.nan_utils import get_same_shape_mask
from ...dispatch import dispatch, evoke

# Import types
from ...transforms.sdes import SDE, LTI_SDE

from .rts_smoother import get_H

import objax
import chex

@jit
def _last_smoothing_element(F, Q, m , P):
    return np.zeros_like(P), m, P

@jit
def _generic_smoothing_element(F, Q, m , P):
    Pp = F @ P @ F.T + Q
    Pp_chol = cholesky(add_jitter(Pp, settings.jitter))
    E = cholesky_solve(Pp_chol, F @ P).T
    g = m - E @ F @ m
    #L = P - E @ F @ P
    L = P - E @ Pp @ E.T

    # FORCE PSD
    # required to fix very small errors that propogate
    L = 0.5 * (L + L.T)
    return E, g, L

@jit
def smoothing_operator(x1, x2):
    # opposite way around as we are in reverse
    # keep indexs this was so it matches the paper
    E_i, g_i, L_i = x2
    E_j, g_j, L_j = x1

    E = E_i @ E_j
    g = E_i @ g_j + g_i
    L = E_i @ L_j @ E_i.T +  L_i


    # FORCE PSD
    # required to fix very small errors that propogate
    L = 0.5 * (L + L.T)

    return E, g, L

@dispatch('parallel')
def smoother(data, prior, filter_res, dt, X_t, X_s, full_state):
    m_init = filter_res['m'][-1]
    P_init = filter_res['P'][-1]

    m_arr = filter_res['m']
    P_arr = filter_res['P']

    # precompute all filtering parameters
    # TODO: this is O(N_t)!! need to distribute
    m_inf = prior.m_inf(None, X_s, None)
    P_inf = prior.P_inf(None, X_s, None)
    H = prior.H(None, X_s, None)
    A_arr = jax.vmap(lambda dt_k: prior.expm(X_s, dt_k))(dt)
    #Q_arr = jax.vmap(lambda A_k: P_inf - A_k @ P_inf @ A_k.T)(A_arr)
    Q_arr = jax.vmap(lambda dt_k, A_k: prior.Q(dt_k, A_k, P_inf, X_s))(dt, A_arr)
    H_arr = np.tile(H[None, ...], [A_arr.shape[0], 1, 1])

    # TODO: check indexes here
    x_all = jax.vmap(
        _generic_smoothing_element
    )(A_arr[:-1], Q_arr[:-1], m_arr[:-1], P_arr[:-1])

    x_last = _last_smoothing_element(None, None, m_arr[-1], P_arr[-1])

    x_all = [
        np.vstack([x_all[i], x_last[i][None, ...]])
        for i in range(3)
    ]

    res = associative_scan(
        jax.vmap(smoothing_operator), 
        x_all,
        reverse=True
    )

    m = res[1]
    P = res[2]

    H_k = get_H(prior, None, None, X_s, X_t[0], full_state)

    # Extract obdereved state
    m = jax.vmap(lambda H_k, m_k: H_k @ m_k)(H_arr, m)
    P = jax.vmap(lambda H_k, P_k: H_k @ P_k @ H_k.T)(H_arr, P)


    return m, P

