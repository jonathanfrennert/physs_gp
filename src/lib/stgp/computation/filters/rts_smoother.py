import jax
from jax import jacfwd, jit
import jax.numpy as np
from jax.lax import scan

from ... import settings 
from ..matrix_ops import cholesky, cholesky_solve, add_jitter
from ..gaussian import log_gaussian, log_gaussian_with_mask
from ...utils.nan_utils import get_same_shape_mask
from ...dispatch import dispatch, evoke

# Import types
from ...transforms.sdes import SDE, LTI_SDE, LinearizedFilter_SDE
from ...transforms.pdes import PDE


import objax
import chex

@dispatch(LinearizedFilter_SDE)
def get_model_H(prior, x, m_predicted, X_s, t, full_state):
    # force full state
    H_k = np.eye(x.shape[0])

    return H_k

@dispatch(LTI_SDE)
def get_model_H(prior, x, m_predicted, X_s, t, full_state):
    if full_state:
        # force full state
        H_k = np.eye(x.shape[0])
    else:
        H_k = prior.H(None, X_s, t)

    return H_k

@dispatch(PDE)
def get_model_H(prior, x, m_predicted, X_s, t, full_state):
    H1 = prior.H(m_predicted, X_s, t) # computed Jacobian at m_predicted 
    if full_state:
        H1 = prior.H_full_state(m_predicted, X_s, t)
    return H1

def get_H(model, x, m_predicted, X_s, t, full_state):
    rts_fn = evoke('get_model_H', model)
    return rts_fn(model, x, m_predicted, X_s, t, full_state)

@jit
def rts_smoother_step(m_filtered_k, P_filtered_k, m, P, m_predicted, P_predicted, A_k, Q_k):
    """
    Computes the RTS smoother step:

         
    """
    P_predicted_chol = cholesky(
        add_jitter(P_predicted, settings.jitter)
    )
    G = cholesky_solve(
        P_predicted_chol, A_k @ P_filtered_k
    ).T

    m = m_filtered_k + G @ (m - m_predicted)
    P = P_filtered_k + G @ (P - P_predicted) @ G.T

    return m, P



@dispatch(LTI_SDE)
def rts_step_wrapper(prior, carry, x, X_s, full_state):
    sde_prior = prior
    P_inf = sde_prior.P_inf(None, X_s, None)

    dt_k = x['dt']

    A_k = sde_prior.expm(X_s, dt_k)
    Q_k = sde_prior.Q(dt_k, A_k, P_inf, X_s)

    m_predicted = A_k @ x['m']
    P_predicted = A_k @ x['P'] @ A_k.T + Q_k

    m, P = rts_smoother_step(
        x['m'],
        x['P'],
        carry['m'],
        carry['P'],
        m_predicted,
        P_predicted,
        A_k, 
        Q_k

    )

    H_k = get_H(prior, x['m'], m_predicted, X_s, x['t'], full_state)


    m_res =  {
        'm': m, 'P': P 
    }

    p_res = {
        'm': H_k @ m, 'P': H_k @ P @ H_k.T
    }


    return m_res, p_res

@dispatch(PDE)
def rts_step_wrapper(prior, carry, x, X_s, full_state):
    sde_prior = prior.parent
    P_inf = sde_prior.P_inf(None, X_s, None)

    dt_k = x['dt']

    A_k = sde_prior.expm(X_s, dt_k)
    Q_k = sde_prior.Q(dt_k, A_k, P_inf, X_s)

    m_predicted = A_k @ x['m']
    P_predicted = A_k @ x['P'] @ A_k.T + Q_k

    m, P = rts_smoother_step(
        x['m'],
        x['P'],
        carry['m'],
        carry['P'],
        m_predicted,
        P_predicted,
        A_k, 
        Q_k

    )


    H_k = get_H(prior, x['m'], m_predicted, X_s, x['t'], full_state)
    #H_k = np.eye(2)


    m_res =  {
        'm': m, 'P': P 
    }

    p_res = {
        'm': H_k @ m, 'P': H_k @ P @ H_k.T
    }

    if settings.debug_mode:
        breakpoint()


    return m_res, p_res

def step_wrapper(data, m, full_state):
    """ Wrapper to support scan with rts_step """

    rts_fn = evoke('rts_step_wrapper', m)

    def _fn(carry, x):
        return rts_fn(m, carry, x, data.X_space, full_state)

    return _fn

@dispatch('sequential')
def smoother(data, model, filter_res, dt, X_t, X_s, full_state):
    m_init = filter_res['m'][-1]
    P_init = filter_res['P'][-1]

    step_wrap = step_wrapper(data, model, full_state)

    carry, ys = scan(
        step_wrap,
        {
            'm': m_init,
            'P': P_init 
        },
        {
            'm': np.flip(filter_res['m'], axis=0)[1:, ...],
            'P': np.flip(filter_res['P'], axis=0)[1:, ...],
            'dt': np.flip(dt, axis=0)[1:, ...],
            't': np.flip(X_t, axis=0)[1:, ...]
        }
    )

    m = ys['m']
    P = ys['P']

    H_k = get_H(model, m_init, m_init, X_s, X_t[0], full_state)
    

    m = np.vstack([(H_k @ m_init)[None, ...], m])
    P = np.vstack([(H_k @ P_init @ H_k.T)[None, ...], P])

    return np.flip(m, axis=0), np.flip(P, axis=0)

def smoother_loop(data: 'SequentialData', model: 'Model', filter_res: dict, full_state=False, filter_type=False):
    """
    Args:
        full_state: flag -- if False we only return part of the state corresponding to the latent GP, else returns the whole state
    """
    # Set up data
    X_t = data.X_time
    X_s =  data.X_space

    N_t = data.Nt
    N_s = data.Ns
    P = data.P

    out_dim = N_s * P

    dt = np.diff(X_t)
    # TODO: fix this
    dt = np.hstack([dt, np.zeros(1)])


    # sequential, parallel, square_root_svm
    smoother_fn = evoke('smoother', filter_type)

    mu, var  =  smoother_fn(data, model, filter_res, dt, X_t, X_s, full_state)

    return mu, var






