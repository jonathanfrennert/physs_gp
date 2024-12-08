import jax
from jax import jit
from functools import partial
import jax.numpy as np
import chex
from typing import List
from objax import ModuleList

from .. import settings
from ..kernels import Kernel, RBF
from ..likelihood import Gaussian
from ..approximate_posteriors import GaussianApproximatePosterior, MM_GaussianInnerLayerApproximatePosterior
from ..dispatch import dispatch
from .gaussian import log_gaussian
from ..sparsity import NoSparsity, Sparsity, FullSparsity
from .. import utils
from .matrix_ops import cholesky, triangular_solve, add_jitter, diagonal_from_cholesky, cholesky_solve, diagonal_from_cholesky, block_diagonal_from_cholesky, get_block_diagonal, block_from_vec, to_block_diag, block_diagonal_from_LXLT

from .permutations import left_permute_mat, permute_vec,permute_vec_ld_to_dl

from .predictors.base_predictors import gaussian_prediction_blocks

@jit
def gaussian_spatial_conditional_diagional(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs_diag, Ktt, m, S_chol, mean_x, mean_xs):
    """
    Computes:
        mu =  Ksz Kzz⁻¹ m
        var =  Ktt * diag[ Kss - Ksz Kzz⁻¹ Kss] + diag[ Ksz Kzz⁻¹ Stt Kzz⁻¹ Kzs ]^T_t
    """

    Kxsxs_diag = np.squeeze(Kxsxs_diag)
    k_zz_chol = cholesky(add_jitter(Kzz, settings.jitter))
    
    A = triangular_solve(k_zz_chol, Kxz.T, lower=True) # M x N
    A1 = triangular_solve(k_zz_chol.T, A, lower=False) # M x N
    A2 = (S_chol.T @ A1).T # N x M 

    mu = mean_xs + A1.T @ (m-mean_x) # N x 1
    sig = Ktt * (Kxsxs_diag - np.sum(np.square(A), axis=0)) + np.sum(np.square(A2), axis=1) #N x 1

    #ensure correct shapes
    mu = np.reshape(mu, [mu.shape[0], 1])
    sig = np.reshape(sig, [sig.shape[0], 1])

    return mu, sig


@jit
def gaussian_spatial_conditional_inv(XS:np.ndarray, X: np.ndarray, Kzz_inv, Kxz, Kxsxs, Ktt, m, S_chol, mean_x, mean_xs):
    """
    Let Kzz = Kzz_inv^{-1}
    Computes:
        mu =  Ksz Kzz⁻¹ m
        var =  Ktt * blkdiag[ Kss - Ksz Kzz⁻¹ Kss] + blkdiag[ Ksz Kzz⁻¹ Stt Kzz⁻¹ Kzs ]^T_t
    """

    N = Kxsxs.shape[0]
    M = m.shape[0]

    chex.assert_shape(Kxsxs, [N, N])
    chex.assert_shape(Kzz_inv, [M, M])
    chex.assert_shape(Kxz, [N, M])
    chex.assert_shape(mean_x, [M, 1])
    chex.assert_shape(mean_xs, [N, 1])
    chex.assert_shape(m, [M, 1])
    chex.assert_shape(S_chol, [M, M])

    A1 = Kzz_inv  @ Kxz.T
    A2 = (S_chol.T @ A1).T # N x M 

    mu = mean_xs + A1.T @ (m-mean_x) # N x 1
    sig = Ktt*(Kxsxs - Kxz @ A1) + A2 @ A2.T #N x N

    #ensure correct shapes
    mu = np.reshape(mu, [N, 1])
    sig = np.reshape(sig, [N, N])

    return mu, sig


@jit
def gaussian_spatial_conditional_cholesky(XS:np.ndarray, X: np.ndarray, Kzz_chol, Kxz, Kxsxs, Ktt, m, S_chol, mean_x, mean_xs):
    """
    Let Kzz = Kzz_chol @ Kzz_chol.T
    Computes:
        mu =  Ksz Kzz⁻¹ m
        var =  Ktt * blkdiag[ Kss - Ksz Kzz⁻¹ Kss] + blkdiag[ Ksz Kzz⁻¹ Stt Kzz⁻¹ Kzs ]^T_t
    """

    N = Kxsxs.shape[0]
    M = m.shape[0]

    chex.assert_shape(Kxsxs, [N, N])
    chex.assert_shape(Kzz_chol, [M, M])
    chex.assert_shape(Kxz, [N, M])
    chex.assert_shape(mean_x, [M, 1])
    chex.assert_shape(mean_xs, [N, 1])
    chex.assert_shape(m, [M, 1])
    chex.assert_shape(S_chol, [M, M])

    A = triangular_solve(Kzz_chol, Kxz.T, lower=True) # M x N
    A1 = triangular_solve(Kzz_chol.T, A, lower=False) # M x N

    A2 = (S_chol.T @ A1).T # N x M 

    mu = mean_xs + A1.T @ (m-mean_x) # N x 1
    sig = Ktt*(Kxsxs - A.T @ A) + A2 @ A2.T #N x N

    #ensure correct shapes
    mu = np.reshape(mu, [N, 1])
    sig = np.reshape(sig, [N, N])

    return mu, sig


@jit
def gaussian_spatial_conditional(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs, Ktt, m, S_chol, mean_x, mean_xs):
    """
    Computes:
        mu =  Ksz Kzz⁻¹ m
        var =  Ktt * blkdiag[ Kss - Ksz Kzz⁻¹ Kss] + blkdiag[ Ksz Kzz⁻¹ Stt Kzz⁻¹ Kzs ]^T_t
    """

    N = Kxsxs.shape[0]
    M = m.shape[0]

    chex.assert_shape(Kxsxs, [N, N])
    chex.assert_shape(Kzz, [M, M])
    chex.assert_shape(Kxz, [N, M])
    chex.assert_shape(mean_x, [M, 1])
    chex.assert_shape(mean_xs, [N, 1])
    chex.assert_shape(m, [M, 1])
    chex.assert_shape(S_chol, [M, M])

    k_zz_chol = cholesky(add_jitter(Kzz, settings.jitter))

    return gaussian_spatial_conditional_cholesky(XS, X, k_zz_chol, Kxz, Kxsxs, Ktt, m, S_chol, mean_x, mean_xs)
   
@jit
def gaussian_linear_operator_spatial_conditional(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs, Ktt, m, S_chol, mean_x, mean_xs):
    """
    Computes:
        mu =  [I_Dt ⊗  Ksz Kzz⁻¹] m_t
        var =  Ktt ⊗  blkdiag[ Kss - Ksz Kzz⁻¹ Kzs] + blkdiag[ [I_Dt ⊗  Ksz Kzz⁻¹] Stt [I_Dt ⊗  Ksz Kzz⁻¹]^T ]^T_t
    """



    # number of dimensions in the time prior
    Dt = Ktt.shape[0]

    N = Kxsxs.shape[0]
    M = m.shape[0]

    #chex.assert_shape(Kxsxs, [N, N])
    #chex.assert_shape(Kzz, [M, M])
    #chex.assert_shape(Kxz, [N, M])
    #chex.assert_shape(mean_x, [M, 1])
    #chex.assert_shape(mean_xs, [N, 1])
    #chex.assert_shape(m, [M, 1])
    #chex.assert_shape(S_chol, [M, M])

    I_t = np.eye(Dt)

    k_zz_chol = cholesky(add_jitter(Kzz, settings.jitter))

    A = triangular_solve(k_zz_chol, Kxz.T, lower=True) # M x N
    A1 = triangular_solve(k_zz_chol.T, A, lower=False) # M x N
    A1_t = np.kron(I_t, A1)

    A2 = (S_chol.T @ A1_t).T # N x M 

    mu = mean_xs + A1_t.T @ (m-mean_x) # N x 1

    sig = np.kron(
        Ktt,
        (Kxsxs - A.T @ A)  #N x N
    ) + A2 @ A2.T



    return mu, sig

@jit
def gaussian_linear_operator_spatial_conditional_no_S_cholesky(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs, Ktt, m, S, mean_x, mean_xs):
    """
    Computes:
        mu =  [I_Dt ⊗  Ksz Kzz⁻¹] m_t
        var =  Ktt ⊗  blkdiag[ Kss - Ksz Kzz⁻¹ Kzs] + blkdiag[ [I_Dt ⊗  Ksz Kzz⁻¹] Stt [I_Dt ⊗  Ksz Kzz⁻¹]^T ]^T_t
    """



    # number of dimensions in the time prior
    Dt = Ktt.shape[0]

    N = Kxsxs.shape[0]
    M = m.shape[0]
    I_t = np.eye(Dt)

    k_zz_chol = cholesky(add_jitter(Kzz, settings.jitter))

    A = triangular_solve(k_zz_chol, Kxz.T, lower=True) # M x N
    A1 = triangular_solve(k_zz_chol.T, A, lower=False) # M x N
    A1_t = np.kron(I_t, A1)

    A2 = (A1_t).T # N x M 

    mu = mean_xs + A1_t.T @ (m-mean_x) # N x 1

    sig = np.kron(
        Ktt,
        (Kxsxs - A.T @ A)  #N x N
    ) + A2 @ S @ A2.T



    return mu, sig

def compute_intermediate_mats(Ktt_q, Kxx_q, Kxz_q, Kzz_q, N, Dt, Ds):
    """
    Args:
        Ktt_q: DtxDt
        Kxx_q: Nsx[Ds]x[Ds]
        Kxz_q: [DsxNs]x[DsxN] or [DsxNs]x[N]
        Kzz_q: [DsxN]x[DsxN] or [N]x[N]

    We want the output to be in Ns-Dt-Ds format. We only permtue X, as it makes no difference 
        if we permute Z or not as they are integrated out.

    NOTE: we have to permute here so that we can work with Kxx_q
    """
    Kxz_up = Kxz_q

    # In [NsxDs]x[DsxN] or [NsxDs]x[N]format
    Kxz_p = left_permute_mat(Kxz_q, Ds)

    # Still in [DsxN]x[DsxN]  or [N]x[N] format
    Kzz_chol = cholesky(add_jitter(Kzz_q, settings.jitter))

    # In [DsxN]x[NsxDs]  or [N]x[NsxDs]
    A = triangular_solve(
        Kzz_chol, 
        Kxz_p.T, 
        lower=True
    )
    # Still in [DsxN]x[NsxDs] or [N]x[N]
    A1 = triangular_solve(Kzz_chol.T, A, lower=False) # M x N


    # Not permutated: In [DsxN]x[DsxNs] or [N]x[DsxNs]
    A_up = triangular_solve(
        Kzz_chol, 
        Kxz_up.T, 
        lower=True
    )

    # Not permutated: In [DsxN]x[DsxNs] [N]x[DsxNs]
    A1_up = triangular_solve(Kzz_chol.T, A_up, lower=False) # M x N

    # Ns x Ds x Ds
    B = block_diagonal_from_cholesky(A.T, Ds)

    # Ns x Ds x Ds
    spatial_pred_var = Kxx_q  - B

    # Ns x [Dt x Ds] x [Dt x Ds]
    spatial_pred_var = jax.vmap(np.kron, [None, 0])(Ktt_q, spatial_pred_var, )

    I_t = np.eye(Dt)

    # Not permutated: [Dt - Ds- Ns] - [Dt - Ds- N]
    A1 = np.kron(I_t, A1_up.T)

    # Not permutated: [Dt - Ds- Ns] - [Dt - Ds- Ns]
    A1_t = np.kron(I_t, A1_up)

    return spatial_pred_var, A1, A1_t

@partial(jit, static_argnums=(0))
def gaussian_linear_operator_spatial_conditional_blocks(block_size: int, XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxx, Ktt, m, S_chol, mean_x, mean_xs):
    """
    Args:
        Ktt: QxDtxDt
        Kzz: Qx[DsxN]x[DsxN] or Qx[Ns]x[N]
        Kxz: Qx[DsxNs]x[DsxN] or Qx[DsxNs]x[N]
        Kxx: NsxQx[Ds]x[Ds]

        m: [Q x Dt x Ds x Ns] x 1 or [Q x Dt x Ns] x 1 
        S_chol: [Q x Dt x Ds x Ns] x [Q x Dt x Ds x Ns] or [Q x Dt x Ns] x [Q x Dt x Ns]
        
    or (depends on hierarchial nor not)

    Output 
        In Ns x [Q x Dt x Ds] format
    """
    # number of dimensions in the time prior
    Q = Ktt.shape[0]
    Dt = Ktt.shape[1]

    # N = number of spatial points
    #Ds = number of spatial outputs
    N, _, Ds, _  = Kxx.shape
    M = m.shape[0]
    
    # We need to compute
    # mu_t = blk_diag[ I_Dt ⊗  Ksz Kzz⁻¹] m_t 
    # var_t = blk_diag[Ktt ⊗  [ Kss - Ksz Kzz⁻¹ Kss]] + blk_diag[[I_Dt ⊗ Ksz Kzz⁻¹]] Stt blk_diag[[I_Dt ⊗ Ksz Kzz⁻¹]^T]

    # Res:
    #   [0] spatial_pred_var: Q x Ns x Ds x Ds
    #   [1] A1: Not permutated: Q x [Dt - Ds- Ns] - [Dt - Ds- N]
    #   [2] A1_t: Not permutated: Q x [Dt - Ds- Ns] - [Dt - Ds- Ns]
    res = jax.vmap(compute_intermediate_mats, [0, 1, 0, 0, None, None, None])(
        Ktt, Kxx, Kxz, Kzz, N, Dt, Ds
    )

    # A1: Not permutated:  [Q x Dt - Ds- Ns] - [Q x Dt - Ds- N] or [Q x Dt - Ds- Ns] - [Q x Dt - N]
    A1_up = to_block_diag(res[1])
    # Not permutated: [Q x Dt - Ds- Ns] - [Q x Dt - Ds- Ns] or [Q x Dt - Ns] - [Q x Dt - Ds- Ns]
    A1_t = to_block_diag(res[2])

    # Ns x Q x [Dt x Ds] x [Dt x Ds]
    spatial_pred_var = np.transpose(res[0], [1, 0, 2, 3])

    # Ns x [Q x Dt x Ds] x [Q x Dt x Ds]
    spatial_pred_var = jax.vmap(to_block_diag)(spatial_pred_var)

    # A1_up @ S_chol: [Q x Dt - Ds- Ns] - [Q x Dt - Ds- N]
    # left_permute_mat -> [Ns x Q x Dt x Ds] - [Q x Dt - Ds- N] 
    B1 = left_permute_mat(A1_up @ S_chol, block_size)

    # Ns x B x B
    # -> Ns x [ Q x Dt x Ds] x [Q x Dt x Ds]
    B1 = block_diagonal_from_cholesky(B1, block_size)

    pred_var = spatial_pred_var + B1

    # [Q x Dt - Ds- Ns] x 1
    pred_mean = A1_t.T @ m 

    # -> [Ns x Q x Dt - Ds] 
    pred_mean = permute_vec_ld_to_dl(pred_mean, num_latents=block_size, num_data=N)
    pred_mean = np.reshape(pred_mean, [N, block_size])

    chex.assert_shape(pred_mean, [N, block_size])
    chex.assert_shape(pred_var, [N, block_size, block_size])

    pred_mean = pred_mean[..., None]
    pred_var = pred_var[:, None, ...]

    return pred_mean, pred_var

@partial(jit, static_argnums=(0))
def gaussian_linear_operator_spatial_conditional_blocks_avoid_S_chol(block_size: int, XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxx, Ktt, m, S, mean_x, mean_xs):
    """
    Args:
        Ktt: QxDtxDt
        Kzz: Qx[DsxN]x[DsxN] or Qx[Ns]x[N]
        Kxz: Qx[DsxNs]x[DsxN] or Qx[DsxNs]x[N]
        Kxx: NsxQx[Ds]x[Ds]

        m: [Q x Dt x Ds x Ns] x 1 or [Q x Dt x Ns] x 1 
        S: [Q x Dt x Ds x Ns] x [Q x Dt x Ds x Ns] or [Q x Dt x Ns] x [Q x Dt x Ns]
        
    or (depends on hierarchial nor not)

    Output 
        In Ns x [Q x Dt x Ds] format
    """
    # number of dimensions in the time prior
    Q = Ktt.shape[0]
    Dt = Ktt.shape[1]

    # N = number of spatial points
    #Ds = number of spatial outputs
    N, _, Ds, _  = Kxx.shape
    M = m.shape[0]
    
    # We need to compute
    # mu_t = blk_diag[ I_Dt ⊗  Ksz Kzz⁻¹] m_t 
    # var_t = blk_diag[Ktt ⊗  [ Kss - Ksz Kzz⁻¹ Kss]] + blk_diag[[I_Dt ⊗ Ksz Kzz⁻¹]] Stt blk_diag[[I_Dt ⊗ Ksz Kzz⁻¹]^T]

    # Res:
    #   [0] spatial_pred_var: Q x Ns x Ds x Ds
    #   [1] A1: Not permutated: Q x [Dt - Ds- Ns] - [Dt - Ds- N]
    #   [2] A1_t: Not permutated: Q x [Dt - Ds- Ns] - [Dt - Ds- Ns]
    res = jax.vmap(compute_intermediate_mats, [0, 1, 0, 0, None, None, None])(
        Ktt, Kxx, Kxz, Kzz, N, Dt, Ds
    )

    # A1: Not permutated:  [Q x Dt - Ds- Ns] - [Q x Dt - Ds- N] or [Q x Dt - Ds- Ns] - [Q x Dt - N]
    A1_up = to_block_diag(res[1])
    # Not permutated: [Q x Dt - Ds- Ns] - [Q x Dt - Ds- Ns] or [Q x Dt - Ns] - [Q x Dt - Ds- Ns]
    A1_t = to_block_diag(res[2])

    # Ns x Q x [Dt x Ds] x [Dt x Ds]
    spatial_pred_var = np.transpose(res[0], [1, 0, 2, 3])

    # Ns x [Q x Dt x Ds] x [Q x Dt x Ds]
    spatial_pred_var = jax.vmap(to_block_diag)(spatial_pred_var)

    # A1_up @ S_chol: [Q x Dt - Ds- Ns] - [Q x Dt - Ds- N]
    # left_permute_mat -> [Ns x Q x Dt x Ds] - [Q x Dt - Ds- N] 
    B1 = left_permute_mat(A1_up, block_size)

    # Ns x B x B
    # -> Ns x [ Q x Dt x Ds] x [Q x Dt x Ds]
    B1 = block_diagonal_from_LXLT(B1, S, block_size)

    pred_var = spatial_pred_var + B1

    # [Q x Dt - Ds- Ns] x 1
    pred_mean = A1_t.T @ m 

    # -> [Ns x Q x Dt - Ds] 
    pred_mean = permute_vec_ld_to_dl(pred_mean, num_latents=block_size, num_data=N)
    pred_mean = np.reshape(pred_mean, [N, block_size])

    chex.assert_shape(pred_mean, [N, block_size])
    chex.assert_shape(pred_var, [N, block_size, block_size])

    pred_mean = pred_mean[..., None]
    pred_var = pred_var[:, None, ...]

    return pred_mean, pred_var

@jit
def gaussian_conditional_diagional(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs_diag, m, S_chol, mean_x, mean_xs) -> np.ndarray:
    """
    Let A = Kxz Kzz⁻¹ then
        
        N(f_s) = ∫ N(f_s | A (f - mean_x) + mean_s, Kxx - A Kzx) N(f | m, S) df
               = N(f_s | mean_s + A (m-mean_x), Kxx - A Kzx + A S A^T)
    """

    return gaussian_spatial_conditional_diagional(
        XS, X, Kzz, Kxz, Kxsxs_diag, 1, m, S_chol, mean_x, mean_xs
    )

@jit
def gaussian_conditional(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs, m, S_chol, mean_x, mean_xs) -> np.ndarray:
    """
    Let A = K(XS, X) K(X, X)^{-1} then
        
        N(f_s) = \int N(f_s | A (f - mean_x) + mean_s, K(XS, XS) - A K(X, XS)) N(f \mid m, S) df
               = N(f_s | mean_s + A (m-mean_x), K(XS, XS) - A K(X, XS) + A S A^T)
    """
    N = Kxsxs.shape[0]
    M = m.shape[0]

    #chex.assert_shape(Kxsxs, [N, N])
    #chex.assert_shape(Kzz, [M, M])
    #chex.assert_shape(Kxz, [N, M])
    #chex.assert_shape(mean_x, [M, 1])
    #chex.assert_shape(mean_xs, [N, 1])
    #chex.assert_shape(m, [M, 1])
    #chex.assert_shape(S_chol, [M, M])

    k_zz_chol = cholesky(add_jitter(Kzz, settings.jitter))
    
    A = triangular_solve(k_zz_chol, Kxz.T, lower=True) # M x N
    A1 = triangular_solve(k_zz_chol.T, A, lower=False) # M x N

    A2 = (S_chol.T @ A1).T # N x M 

    mu = mean_xs + A1.T @ (m-mean_x) # N x 1
    sig = Kxsxs - A.T @ A + A2 @ A2.T #N x N

    #ensure correct shapes
    mu = np.reshape(mu, [N, 1])
    sig = np.reshape(sig, [N, N])

    return mu, sig

@partial(jit, static_argnums=(0, 1))
def gaussian_conditional_blocks(group_size, block_size, XS:np.ndarray, X: np.ndarray, Kzz, K_xz, Kxx, m, S_chol, mean_x, mean_xs) -> np.ndarray:
    """
    We want to compute
        blk_diag(Kxz Kzz^{-1} m), 
        blk_diag(Kxx - Kxz Kzz^{-1} Kzx + Kxz Kzz^{-1} S  Kzz^{-1} Kzx)

    We rewrite the variance as 

        blk_diag(Kxx - Kxz Kzz^{-1} Kzx)  + blk_diag (Kxz Kzz^{-1} S  Kzz^{-1} Kzx) = R1 + R2

    We can compute R1 using gaussian_prediction_blocks. R2 is computed as:
        
        R2 = blk_diag_cholesky_product(Kxz Kzz^{-1}) S^{1/2})
    """

    if settings.safe_mode:
        NS  = XS.shape[0]
        N = X.shape[0]
        Q = block_size
        # slow but is jittable
        mu, var = gaussian_conditional(XS, X, Kzz, K_xz, np.zeros([NS * Q, NS * Q]), m, S_chol, np.zeros_like(m), np.zeros([NS * Q, 1]))

        var = Kxx + get_block_diagonal(var, Q)
                     
        mu = block_from_vec(mu, block_size)

        return mu, var

    #TODO: THERE IS A JIT BUG IN HERE SOMEWHERE??
    M = X.shape[1]
    N = XS.shape[0]
    Q = block_size

    chex.assert_shape(Kxx, [N, Q, Q])
    chex.assert_equal(Kzz.shape[0], Kzz.shape[1])
    chex.assert_equal(K_xz.shape[0], N*Q)
    chex.assert_equal(K_xz.shape[1], Kzz.shape[0])
    chex.assert_shape(m, [Kzz.shape[0], 1])
    chex.assert_shape(S_chol, Kzz.shape)

    # Add jitter to help the cholesy solve
    jit_arr = np.eye(Kzz.shape[0])*settings.jitter

    # pred_mu = Kxz(Kzz+jit)^{-1}m
    # pred_var = Kxx - Kxz(Kzz+jit)^{-1}Kxz.T
    pred_mu, pred_var = gaussian_prediction_blocks(
        group_size, block_size, m,  Kxx, K_xz, Kzz, mean_x, mean_xs, jit_arr
    )

    chex.assert_shape(pred_mu, [N, Q])
    chex.assert_shape(pred_var, [N, Q, Q])

    # Compute KxzKzz^{-1}SKzz^{-1}Kxz.T
    K_chol = cholesky(add_jitter(Kzz, settings.jitter))

    A = cholesky_solve(K_chol, K_xz.T)

    A2 = S_chol.T @ A 

    B = block_diagonal_from_cholesky(A2.T, block_size)

    mu = pred_mu
    var =  pred_var + B

    return mu, var

@jit
def gaussian_conditional_covar(X1:np.ndarray, X2:np.ndarray, X: np.ndarray, Kzz, Kxz, Kzx, Kxsxs, m, S_chol) -> np.ndarray:
    k_zz_chol = cholesky(add_jitter(Kzz, settings.jitter))

    A1 = cholesky_solve(k_zz_chol, Kzx)
    sig1 = Kxsxs - Kxz @ A1
    sig2 = Kxz @ cholesky_solve(k_zz_chol, S_chol) @ S_chol.T @ A1

    sig = sig1 + sig2

    return sig

@jit
def whitened_gaussian_conditional_diagional(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs_diag, m, S_chol) -> np.ndarray:
    """
        Args:
            g1 defines the distribution of the conditional p(f*|f)
            g2 defines the distribution that the expectation is wrt E_{q(f)} [ . ]

    """

    k_zz = Kzz
    k_xz = Kxz
    k_xx_diag = Kxsxs_diag
    mu = m
    sig_chol = S_chol

    chex.assert_rank(k_zz, 2)
    chex.assert_rank(k_xz, 2)
    chex.assert_rank(k_xx_diag, 1)
    chex.assert_rank(mu, 2)
    chex.assert_rank(sig_chol, 2)

    k_zz_chol = cholesky(add_jitter(k_zz, settings.jitter))

    mu = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, mu, lower=False)

    A1 = jax.scipy.linalg.solve_triangular(k_zz_chol, k_xz.T, lower=True)
    A2 = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, sig_chol, lower=False)

    sig = k_xx_diag - np.sum(np.square(A1), axis=0) + np.sum(np.square(A2), axis=1)

    mu = np.reshape(mu, [mu.shape[0], 1])
    sig = np.reshape(sig, [sig.shape[0], 1])

    return mu, sig

@jit
def whitened_gaussian_conditional_full(XS:np.ndarray, X: np.ndarray, Kzz, Kxz, Kxsxs, m, S_chol) -> np.ndarray:
    """
        Args:
            g1 defines the distribution of the conditional p(f*|f)
            g2 defines the distribution that the expectation is wrt E_{q(f)} [ . ]
    """
    k_zz = Kzz
    k_xz = Kxz
    k_xsxs = Kxsxs
    mu = m
    sig_chol = S_chol
    sig = sig_chol @ sig_chol.T

    k_zz_chol = cholesky(add_jitter(k_zz, settings.jitter))


    A = cholesky_solve(k_zz_chol, k_xz.T)
    A1 = jax.scipy.linalg.solve_triangular(k_zz_chol, k_xz.T, lower=True)

    mu = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, mu, lower=False)

    sig = k_xsxs - k_xz @ A + A1.T @ sig @ A1

    return mu, sig


@jit
def whitened_gaussian_covar(X1:np.ndarray, X2:np.ndarray, X: np.ndarray, Kzz, Kxz, Kzx, Kxsxs, m, S_chol) -> np.ndarray:
    """
        Args:
            g1 defines the distribution of the conditional p(f*|f)
            g2 defines the distribution that the expectation is wrt E_{q(f)} [ . ]
    """
    sig_chol = S_chol
    sig = sig_chol @ sig_chol.T

    k_zz_chol = cholesky(add_jitter(k_zz, settings.jitter))
    A = cholesky_solve(k_zz_chol, Kzx)

    mu = k_xz @ jax.scipy.linalg.solve_triangular(k_zz_chol.T, mu, lower=False)

    sig = k_xsxs - kxz @ A + cholesky_solve(k_zz_chol, Kxz.T) @ sig @ A


    return mu, sig


