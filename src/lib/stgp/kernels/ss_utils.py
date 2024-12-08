import jax
import jax.numpy as np
from jax import jit

# matern32 functionals
@jit
def matern32_temporal_expm(dt, lengthscales):
    lam = np.sqrt(3.0) / lengthscales
    A = np.exp(-dt * lam) * (dt * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2))
    return A

@jit 
def matern32_temporal_state_space_rep(lengthscale, variance):
    # temporal so input dim in 1
    v = 3.0 / 2.0
    D = int(v + 0.5)

    lam = (3.0 ** 0.5) / lengthscale
    F = np.array([[0.0, 1.0], [-(lam ** 2), -2 * lam]])

    L = np.array([[0.0], [1.0]])

    # measurement model matrix
    H = np.array([[1.0, 0.0]])

    Qc = np.array([
        [12.0 * 3.0 ** 0.5 / lengthscale ** 3.0 * variance]
    ])

    minf = np.zeros([2, 1])
    Pinf = np.array(
        [
            [variance, 0.0],
            [0.0, 3.0 * variance / lengthscale ** 2.0],
        ]
    )

    return F, L, Qc, H, minf, Pinf


@jit
def space_time_state_space_rep(K_spatial, F, L, Qc, H, m_inf, Pinf):
    Ns = K_spatial.shape[0]
    eye = np.eye(Ns)

    F_st = np.kron(eye, F)
    L_st = np.kron(eye, L)
    Qc_st = np.kron(K_spatial, Qc)
    H_st = np.kron(eye, H)
    Pinf_st = np.kron(K_spatial, Pinf)
    m_inf_st = np.kron(np.ones([Ns, 1]), m_inf)

    return F_st, L_st, Qc_st, H_st, m_inf_st, Pinf_st

