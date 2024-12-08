import jax
from jax import jacfwd, jit
import jax.numpy as np
from jax.lax import scan, associative_scan

from ... import settings 
from ..matrix_ops import cholesky, cholesky_solve, add_jitter, mat_inv, force_symmetric, solve_with_additive_inverse, vec_add_jitter, triangular_solve, log_chol_matrix_det
from ..gaussian import log_gaussian, log_gaussian_with_mask, log_gaussian_with_additive_precision_noise_with_mask
from ...utils.nan_utils import get_same_shape_mask
from ...dispatch import dispatch, evoke

from jax.numpy.linalg import svd

# Import types
from ...transforms.sdes import SDE, LTI_SDE

from ..linalg import solve, solve_from_cholesky

import jax.scipy as jsp

import objax
import chex

def get_data_mask(Y_k):
    mask_k = get_same_shape_mask(Y_k)

    # Construct spatial mask
    m_vec = np.tile(mask_k, [1, Y_k.shape[0]])

    M = np.multiply(
        m_vec,
        np.eye(Y_k.shape[0])
    )

    return M

@jit
def psd_sqrt(A):
    A_chol = cholesky(add_jitter(A, settings.jitter))
    A_U, A_D, _ = svd(A_chol, full_matrices=False)

    return A_U @ np.diag(A_D)

@jit
def svd_sqrt(A_sqrt):
    A_sqrt_U, A_sqrt_D, A_sqrt_VT  = svd(A_sqrt, full_matrices=False)
    A_sqrt_D = np.diag(A_sqrt_D)

    return A_sqrt_U @ A_sqrt_D, A_sqrt_U, A_sqrt_D, A_sqrt_VT

@jit
def lml(Y_k, R_chol, R_chol_inv_y, R_chol_inv_MH, U, D, m_):
    N = Y_k.shape[0]

    c1 = -(N/2)*np.log(2*np.pi)

    DI = np.diag(D**2 + np.eye(D.shape[0]))
    # log dets
    log_det =  log_chol_matrix_det(R_chol) + np.sum(np.log(DI))

    mahal = np.diag((1/np.sqrt(DI))) @ U.T @ (R_chol_inv_y - R_chol_inv_MH @ m_)

    res = c1 + -0.5 * log_det - 0.5*mahal.T @ mahal
    return np.sum(c1 + -0.5 * log_det - 0.5*mahal.T @ mahal)

@jit
def sr_svm_kf_step(carry, x):
    m_k = carry['m']
    P_sqrt_k = carry['P_sqrt']

    dt_k = x['dt']
    Y = x['Y'] 
    lik_mat = x['lik_mat'] 
    t = x['t'] 
    A = x['A'] 
    Q_sqrt = x['Q_sqrt'] 
    H = x['H'] 

    R_chol = x['R_chol'] 
    R_chol_inv_y = x['R_chol_inv_y'] 
    R_chol_inv_MH = x['R_chol_inv_MH'] 

    # set up

    mask_k = get_same_shape_mask(Y)

    Y = np.nan_to_num(Y)

    # Construct spatial mask
    m_vec = np.tile(mask_k, [1, Y.shape[0]])

    M = np.multiply(
        m_vec,
        np.eye(Y.shape[0])
    )

    # only observe on nan
    H_k = M @ H

    # predict step

    # D x 2D
    m_ = A @ m_k
    P_sqrt_ = np.block([
        A @ P_sqrt_k,
        Q_sqrt
    ])

    P_sqrt_, _, _, _ = svd_sqrt(P_sqrt_)

    # update step
    mu = H_k @ m_

    #inovation mean and variance
    # all in latent-space format
    v = Y - mu

    _, U, D, VT = svd_sqrt(
        R_chol_inv_MH @  P_sqrt_
    )

    log_Z_k = lml(Y, R_chol, R_chol_inv_y, R_chol_inv_MH, U, D, m_)

    I = np.eye(D.shape[0])

    K_tilde = VT.T @ D @ np.diag((1/np.sqrt(np.diag(D**2 + I))))

    _, U_tilde, D_tilde, VT_tilde = svd_sqrt(K_tilde)

    K_gain_no_R = P_sqrt_ @ VT.T @ D @ np.diag((1/np.diag((D**2 + I)))) @ U.T
    m_k = m_ + K_gain_no_R @ (R_chol_inv_y - R_chol_inv_MH @ m_)

    I = np.eye(K_gain_no_R.shape[0])

    # (I - K_gain_no_R @ R_chol_inv_MH) @ P_sqrt_ @ ((I - K_gain_no_R @ R_chol_inv_MH) @ P_sqrt_).T + K_gain_no_R@K_gain_no_R.T
    P_k_sqrt = np.block([ (I - K_gain_no_R @ R_chol_inv_MH) @ P_sqrt_, K_gain_no_R ])
    P_k_sqrt, _, _, _ = svd_sqrt(P_k_sqrt)

    return {
        'm': m_k, 'P_sqrt': P_k_sqrt 
    }, {
        'm': m_k, 'P_sqrt': P_k_sqrt, 'lml': log_Z_k
    }




@dispatch('square_root_svm')
def filter(data, prior, lik_mat, Y, X_t, X_s, dt, lik_cov_flag):
    # TODO: ASSUMING LTI_SDE

    # compute steady states
    P_inf = prior.P_inf(None, X_s, None)
    m_inf = prior.m_inf(None, X_s, None)
    H = prior.H(None, X_s, None)

    P_inf_sqrt = psd_sqrt(P_inf) 

    lik_mat_arr = lik_mat
    A_arr = jax.vmap(lambda dt_k: prior.expm(X_s, dt_k))(dt)
    Q_arr = jax.vmap(lambda A_k: P_inf - A_k @ P_inf @ A_k.T)(A_arr)
    H_arr = np.tile(H[None, ...], [lik_mat_arr.shape[0], 1, 1])

    # TODO - this could get quite big!
    Q_sqrt_arr = jax.vmap(psd_sqrt)(Q_arr)

    # precompute all cholesky based operations so they are
    # not inside of the scan

    lik_mat_jitted = vec_add_jitter(lik_mat, settings.jitter)

    # precompute data masks as the observation matrix will actuall be
    # M @ H
    M_arr = jax.vmap(get_data_mask)(Y)

    MH_arr = jax.vmap(lambda M, H: M @ H)(M_arr, H_arr)

    R_chol = jax.vmap(cholesky)(lik_mat_jitted)
    R_chol_inv_y = jax.vmap(triangular_solve)(R_chol, np.nan_to_num(Y))
    R_chol_inv_MH = jax.vmap(triangular_solve)(R_chol, MH_arr)


    carry, ys = scan(
        sr_svm_kf_step,
        {
            'm': m_inf,
            'P_sqrt': P_inf_sqrt
        },
        {
            'dt': dt,
            't': X_t,
            'Y': Y,
            'lik_mat': lik_mat,
            'A': A_arr,
            'Q_sqrt': Q_sqrt_arr,
            'H': H_arr,
            'R_chol': R_chol,
            'R_chol_inv_y': R_chol_inv_y,
            'R_chol_inv_MH': R_chol_inv_MH,
        }
    )


    lml = np.sum(ys['lml'])

    P = jax.vmap(lambda a: a @ a.T)(ys['P_sqrt'])

    filter_res = {'m': ys['m'], 'P': P}

    return lml, filter_res
