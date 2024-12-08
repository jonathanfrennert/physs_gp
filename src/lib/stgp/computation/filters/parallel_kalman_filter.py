""" 
Closely following 
    https://github.com/EEA-sensors/parallel-gps/blob/main/pssgp/kalman/parallel.py
    https://github.com/AaltoML/BayesNewton/blob/61cb0ebb23afb12de0008882bc3d16b864b7149e/bayesnewton/ops.py#L521
    https://github.com/tensorflow/probability/blob/399cfcb4edda192c9f0f070ac04035dcb0e5b3a5/tensorflow_probability/python/experimental/parallel_filter/parallel_kalman_filter_lib.py#L701
"""
import jax
from jax import jacfwd, jit
import jax.numpy as np
from jax.lax import scan, associative_scan

#import tensorflow_probability

from ... import settings 
from ..matrix_ops import cholesky, cholesky_solve, add_jitter, mat_inv, force_symmetric, solve_with_additive_inverse
from ..gaussian import log_gaussian, log_gaussian_with_mask, log_gaussian_with_additive_precision_noise_with_mask
from ...utils.nan_utils import get_same_shape_mask
from ...dispatch import dispatch, evoke

# Import types
from ...transforms.sdes import SDE, LTI_SDE

from ..linalg import solve, solve_from_cholesky

import jax.scipy as jsp

import objax
import chex

@jit
def fix_psd(A):
    return force_symmetric(A)

@jit
def _first_filtering_element_lik_precision(m, P, F, Q, H, R_inv, y):
    m_ = F @ m
    P_ = F @ P @ F.T + Q

    # S = H P_ H.T + R
    #   = T1 + R
    T1 =  H @ P_ @ H.T
    T = T1 @ R_inv + np.eye(T1.shape[0])

    R_inv_chol = cholesky(add_jitter(R_inv, settings.jitter))

    # K = P_ H.T S_inv
    #   =  (S_inv @ H @ P_).T
    #   =  ([H P_ H.T + R]^{-1} @ H @ P_).T
    K = solve_with_additive_inverse(T1, R_inv, H @ P_.T).T

    A = np.zeros_like(F)

    # b = m_ + K @ [y-Hm_]
    b = m_ + K @ (y - H @ m_)

    # C = P_ - K @ S @ K.T
    C = P_ - K @ T @ cholesky_solve(R_inv_chol,  K.T)

    # S_k - H Q H.T + R
    # eta = F.T H.T S_k^{-1} (y)
    # J = F.T H.T S_k^{-1} H F

    FH_S_inv = solve_with_additive_inverse(H @ Q @ H.T, R_inv, H @ F).T

    eta = FH_S_inv @ y
    J = FH_S_inv @ H @ F 

    C = force_symmetric(C)
    J = force_symmetric(J)

    return A, b, C, J, eta

@jit
def _first_filtering_element(m, P, F, Q, H, R, y):
    m_ = F @ m
    P_ = F @ P @ F.T + Q


    S1 = H @ P_ @ H.T + R
    #S1_chol = cholesky(S1)
    #K = cholesky_solve(S1_chol, H @ P_.T).T
    K = solve(S1, H @ P_.T).T

    A = np.zeros_like(F)
    b = m_ + K @ (y - H @ m_)
    C = P_ - K @ S1 @ K.T

    S = H @ Q @ H.T + R

    #S_chol = cholesky(S)
    #FH_S_inv = cholesky_solve(S_chol, H @ F).T
    FH_S_inv = solve(S, H @ F).T

    eta = FH_S_inv @ y
    J = FH_S_inv @ H @ F 

    C = force_symmetric(C)
    J = force_symmetric(J)

    return A, b, C, J, eta

@jit
def _first_filtering_element_nan(m, P, F, Q, H, R, y):
     
    A = np.zeros_like(F)
    b = m
    C = P
    eta = np.zeros_like(m)
    J = np.zeros_like(F)

    return A, b, C, J, eta

# Does not depend on R
_first_filtering_element_nan_lik_precision = _first_filtering_element_nan


@jit
def _generic_filtering_element_lik_precision(F, Q, H, R_inv, y):
    I = np.eye(F.shape[0])

    # S_k = H Q H.T + R
    # S_k = T1 + R
    T1 =  H @ Q @ H.T
    T = T1 @ R_inv + np.eye(T1.shape[0])

    # K = Q H.T S^{-1}
    K = solve_with_additive_inverse(T1, R_inv, H @ Q.T).T

    A = (I - K @ H) @ F
    b = K @ y
    C = (I - K @ H) @ Q

    # eta = F.T H.T S_k^{-1} (y)
    # J = F.T H.T S_k^{-1} H F

    FH_S_inv = solve_with_additive_inverse(H @ Q @ H.T, R_inv, H @ F).T

    eta = FH_S_inv @ y
    J = FH_S_inv @ H @ F 

    return A, b, C, J, eta

@jit
def _generic_filtering_element(F, Q, H, R, y):
    I = np.eye(F.shape[0])


    S = H @ Q @ H.T + R

    #S_chol = cholesky(S)
    #K = cholesky_solve(S_chol, H @ Q.T).T
    K = solve(S, H @ Q.T).T

    A = (I - K @ H) @ F
    b = K @ y
    C = (I - K @ H) @ Q
    #eta = F.T @ H.T @ cholesky_solve(S_chol, y)
    #J = F.T @ H.T @  cholesky_solve(S_chol, H @ F) 
    eta = F.T @ H.T @ solve(S, y)
    J = F.T @ H.T @  solve(S, H @ F) 

    return A, b, C, J, eta

@jit
def _generic_filtering_element_nan(F, Q, H, R, y):
    A = F
    b = np.zeros([F.shape[1], 1])
    C = Q
    J = np.zeros_like(Q)
    eta = np.zeros_like(b)

    return A, b, C, J, eta

# does not depend on R
_generic_filtering_element_nan_lik_precision = _generic_filtering_element_nan


@jit
def filtering_operator(x1, x2):
    """ combine individual elements """
    A_i, b_i, C_i, J_i, eta_i = x1
    A_j, b_j, C_j, J_j, eta_j = x2

    N, D = A_i.shape
    I = np.eye(D)

    if settings.parallel_kf_force_linear_solve:
        # scarifice some stability to be able to use CG and cholesky solves
        C_i_inv = solve(C_i, np.eye(C_i.shape[0]))
        inner_tmp = C_i_inv + J_j
        Aj_tmp = (solve(inner_tmp, A_j.T).T) @ C_i_inv

        A = Aj_tmp @ A_i
        C = Aj_tmp @ C_i @ A_j.T + C_j
        b = Aj_tmp @ (b_i + C_i @ eta_j)+b_j


        A_i_tmp = solve(inner_tmp.T, C_i_inv @ A_i).T

    else:
        inner_tmp = I+C_i@J_j
        #Aj_tmp = np.linalg.solve(inner_tmp.T, A_j.T).T
        Aj_tmp = jsp.linalg.solve(inner_tmp.T, A_j.T, assume_a='gen').T

        A = Aj_tmp @ A_i
        C = Aj_tmp @ C_i @ A_j.T + C_j
        b = Aj_tmp @ (b_i + C_i @ eta_j)+b_j

        inner_tmp = I+J_j@C_i
        #A_i_tmp = np.linalg.solve(inner_tmp.T, A_i).T
        A_i_tmp = jsp.linalg.solve(inner_tmp.T, A_i, assume_a='gen').T

    eta = A_i_tmp @ (eta_j - J_j @ b_i) + eta_i
    J = A_i_tmp @ J_j @ A_i + J_i

    # FORCE PSD
    # required to fix very small errors that propogate
    C = fix_psd(C) 
    J = fix_psd(J) 
    return A, b, C, J, eta

def make_filtering_elements():
    pass

@dispatch('parallel')
def filter(data, prior, lik_mat, Y, X_t, X_s, dt, lik_cov_flag, train_test_mask, train_index):

    # compute steady states
    P_inf = prior.P_inf(None, X_s, None)
    m_inf = prior.m_inf(None, X_s, None)
    H = prior.H(None, X_s, None)

    # precompute all filtering parameters
    # TODO: this is O(N_t)!! need to distribute
    #dt = np.ones(Y.shape[0])*dt[1]
    lik_mat_arr = lik_mat
    A_arr = jax.vmap(lambda dt_k: prior.expm(X_s, dt_k))(dt)
    #Q_arr = jax.vmap(lambda A_k: P_inf - A_k @ P_inf @ A_k.T)(A_arr)
    Q_arr = jax.vmap(lambda dt_k, A_k: prior.Q(dt_k, A_k, P_inf, X_s))(dt, A_arr)
    H_arr = np.tile(H[None, ...], [lik_mat_arr.shape[0], 1, 1])
    #Q_arr = Q_arr.at[0].set(P_inf)

    mask = get_same_shape_mask(Y)
    # collapse mask
    mask = np.any(np.any(mask, axis=1), axis=1).astype(int)

    Y = np.nan_to_num(Y)

    if lik_cov_flag:
        # lik_mat is a covariance
        x_0 = _first_filtering_element(m_inf, P_inf, A_arr[0], P_inf, H, lik_mat_arr[0], Y[0])
        x_0_nan = _first_filtering_element_nan(m_inf, P_inf, A_arr[0], P_inf, H, lik_mat_arr[0], Y[0])
    else:
        # lik_mat is a precision
        x_0 = _first_filtering_element_lik_precision(m_inf, P_inf, A_arr[0], P_inf, H, lik_mat_arr[0], Y[0])
        x_0_nan = _first_filtering_element_nan_lik_precision(m_inf, P_inf, A_arr[0], P_inf, H, lik_mat_arr[0], Y[0])

    # combine nan and observered
    x_0 = [
        x_0[i] * mask[0] + x_0_nan[i] * (1-mask[0])
        for i in range(5)
    ]

    if lik_cov_flag:
        x_all = jax.vmap(
            _generic_filtering_element
        )(A_arr, Q_arr, H_arr, lik_mat_arr, Y)

        x_all_nan = jax.vmap(
            _generic_filtering_element_nan
        )(A_arr, Q_arr, H_arr, lik_mat_arr, Y)
    else:
        x_all = jax.vmap(
            _generic_filtering_element_lik_precision
        )(A_arr, Q_arr, H_arr, lik_mat_arr, Y)

        x_all_nan = jax.vmap(
            _generic_filtering_element_nan_lik_precision
        )(A_arr, Q_arr, H_arr, lik_mat_arr, Y)

    # combine nan and observered
    def get_mask(mask, r):
        return np.reshape(mask, [-1] + [1]*(len(r.shape)-1))

    x_all = [
        x_all[i] * get_mask(mask, x_all[i]) + x_all_nan[i] * (1-get_mask(mask, x_all[i]))
        for i in range(5)
    ]

    x_all = [
        x_all[i].at[0].set(x_0[i])
        for i in range(5)
    ]

    res = associative_scan(
        jax.vmap(filtering_operator), 
        x_all
    )

    filtered_means = np.vstack([m_inf[None, ...], res[1][:-1]])
    filtered_cov = np.vstack([P_inf[None, ...], res[2][:-1]])

    obs_means = jax.vmap(
        lambda H_k, m_k, F_k: H_k @ F_k @ m_k
    ) (
        H_arr, filtered_means, A_arr
    ) 
    obs_pred_cov = jax.vmap(
        lambda H_k, P_k, F_k, Q_k: H_k @ F_k @ P_k @ F_k.T @ H_k.T  + H_k @ Q_k @ H_k.T
    )(H_arr, filtered_cov, A_arr, Q_arr)


    if lik_cov_flag:
        log_Z_k = jax.vmap(
            lambda Y_k, mu_k, S_k, R_k: np.sum(
                log_gaussian_with_mask(np.nan_to_num(Y_k), mu_k, S_k+R_k, get_same_shape_mask(Y_k)[:, 0])
            )
        ) (Y, obs_means, obs_pred_cov, lik_mat_arr)
    else:
        log_Z_k = jax.vmap(
            lambda Y_k, mu_k, S_k, R_inv_k: np.sum(
                log_gaussian_with_additive_precision_noise_with_mask(
                    np.nan_to_num(Y_k), mu_k, S_k, R_inv_k, get_same_shape_mask(Y_k)[:, 0]
                )
            )
        ) (Y, obs_means, obs_pred_cov, lik_mat_arr)

        tmp_log_Z_k = jax.vmap(
            lambda Y_k, mu_k, S_k, R_k: np.sum(
                log_gaussian_with_mask(np.nan_to_num(Y_k), mu_k, S_k+R_k, get_same_shape_mask(Y_k)[:, 0])
            )
        ) (Y, obs_means, obs_pred_cov, jax.vmap(mat_inv)(lik_mat_arr))
        
    log_Z = np.sum(log_Z_k)

    return log_Z, {'m': res[1], 'P': res[2]}

