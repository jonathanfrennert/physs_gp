"""Standard operations on matrices."""

import jax
import jax.numpy as np
from jax.scipy.sparse.linalg import cg
from jax import jit, grad
from functools import partial
import chex
from jax import jacfwd, jacrev, grad
import objax
from .. import settings
from jax.scipy.linalg import expm


def first_axis_dim(X):
    if type(X) is list:
        return len(X)
    return X.shape[0]

def shape_rank(X):
    if type(X) is objax.module.ModuleList:
        return len(X)

    if type(X) is list:
        # min x
        return 1 + np.min(np.array([shape_rank(x) for x in X]))

    if hasattr(X, "shape"):
        shape = X.shape
    else:
        shape = ()  
    return len(shape)



#@partial(jit, static_argnums=(1))
def hessian(f, argnums):
    return jacfwd(jacrev(f, argnums=argnums), argnums=argnums)

@jit
def to_block_diag(A):
    if True:
        A = np.array(A)
        blocks = A.shape[0]
        A_stacked = np.vstack(A)

        A_full = np.tile(A_stacked, [1, blocks])
        mask = np.kron(np.eye(blocks), np.ones_like(A[0]))
        res_1 =  A_full * mask
        return res_1
    else:
        res =  jax.scipy.linalg.block_diag(*A)
        return res

@partial(jit, static_argnums=(1))
def get_block_diagonal(A, block_size):
    chex.assert_rank(A, 2)

    if False:
        N = A.shape[0]
        num_blocks = int(N / block_size)
        
        
        A_1 = A.reshape(num_blocks, block_size, block_size*num_blocks)
        A_2 = np.transpose(A_1, [0, 2, 1])
        A_3 = np.reshape(A_2, [num_blocks, num_blocks, block_size*block_size])
        
        A_4 = jax.vmap(lambda a, i: a[i])(A_3, np.arange(num_blocks))

        A_5 = np.reshape(A_4, [num_blocks, block_size, block_size])
        A_6 = np.transpose(A_5, [0, 2, 1])
        return A_6
    else:
        N = A.shape[0]

        num_blocks = N / block_size
        a = np.array(list(range(N)))

        indexes = np.reshape(a, [-1, block_size])

        blocks =  jax.vmap(
            lambda A, i: A[i,:][:, i],
            [None, 0],
            0
        )(A, indexes)

    chex.assert_shape(blocks, [num_blocks, block_size, block_size])

    return blocks

@partial(jit, static_argnums=(1))
def batched_block_diagional(A, block_size):
    chex.assert_rank(A, 3)
    return jax.vmap(get_block_diagonal, [0, None])(A, block_size)


@jit
def batched_diag(A):
    chex.assert_rank(A, 2)
    return jax.vmap(np.diag, [0])(A)


@jit
def cartesian_product(X, Y):
    return np.vstack([np.tile(X, Y.shape[0]), np.repeat(Y, X.shape[0])])

@jit
def add_jitter(A, jit_val):
    chex.assert_rank(A, 2)
    return A + jit_val*np.eye(A.shape[0])

@jit
def vec_add_jitter(A, jit_val):
    chex.assert_rank(A, 3)

    return jax.vmap(
        add_jitter,
        (0, None),
        0
    )(A, jit_val)

@jit
def diagonal_from_cholesky(L):
    """ Compute diag(LL^T). """
    # ensure square matrix
    chex.assert_rank(L, 2)
    chex.assert_equal(L.shape[0], L.shape[1])

    diag = np.sum(np.square(L), axis=1)
    diag = np.reshape(diag, [L.shape[0], 1])

    return diag


@jit
def diagonal_from_XDXT(X, d):
    """
    Computes diagonal of XDX^T where D is a diagonal matrix

    This is done by forming the matrix square root:
        L = X @ diag(sqrt(d))

    and calling diagonal_from_cholesky(L)

    Returns:
        diag(LL^T): N x 1
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(d, 1)

    return diagonal_from_cholesky(X @ np.sqrt(np.diag(d)))

@jit
def batched_diagonal_from_XDXT(X, D):
    """
    Batches diagonal_from_XDXT over the second arg D

    Returns the same shape as D
    """
    chex.assert_rank(X, 2)
    chex.assert_rank(D, 2)

    res =  jax.vmap(diagonal_from_XDXT, [None, 1])(X, D)

    res = res[..., 0].T

    chex.assert_equal_shape([res, D])

    return res

@partial(jit, static_argnums=(2))
def block_diagonal_from_LXLT(L, X, block_size):
    """
    Extracts block diagonals from LX L^T
    """
    # TODO: implement fast version
    R = L @ X @ L.T

    return get_block_diagonal(R, block_size)

@partial(jit, static_argnums=(1))
def block_diagonal_from_cholesky(L, block_size):
    """
    Extracts block diagonals from L L^T
    """

    # TODO: clean up
    if False:
        R = L @ L.T

        return get_block_diagonal(R, block_size)

    L1 = np.reshape(
        L, 
        [-1, block_size, L.shape[-1]]
    )

    B = L1 @ np.transpose(L1, [0, 2, 1])

    return B

@partial(jit, static_argnums=(1))
def block_from_vec(x, block_size):
    return np.reshape(x, [-1, block_size])

@partial(jit, static_argnums=(1))
def block_from_mat(X, block_size):
    return np.reshape(X, [-1, block_size, X.shape[1]])

@jit
def log_chol_matrix_det(chol):
    # ensure square matrix
    chex.assert_rank(chol, 2)
    chex.assert_equal(chol.shape[0], chol.shape[1])

    val = np.square(np.diag(chol))
    return np.sum(np.log(val))


@jit
def cholesky_solve(chol, X):
    # ensure square matrix
    chex.assert_rank(chol, 2)
    chex.assert_equal(chol.shape[0], chol.shape[1])

    # ensure shapes conform
    chex.assert_equal(chol.shape[1], X.shape[0])

    # assumes chol is lower
    return jax.scipy.linalg.cho_solve([chol, True], X)


@jit
def cholesky(A):
    # return lower triangular cholesky factor
    return np.linalg.cholesky(A)

@jit
def batched_cholesky(A):
    chex.assert_rank(A, 3)
    return jax.vmap(cholesky, 0)(A)

@partial(jit, static_argnums=(2))
def _triangular_solve(chol, X, lower):
    return jax.scipy.linalg.solve_triangular(chol, X, lower=lower) 

def triangular_solve(chol, X, lower=True):
    #wrapper around _triangular_solve so that lower can be a keyword arg
    return _triangular_solve(chol, X, lower)

@jit
def to_lower_triangular_vec(A):
    N = A.shape[0]
    return A[np.tril_indices(N)].flatten()

@jit
def lower_triangular_cholesky(A):
    return to_lower_triangular_vec(cholesky(A))



@jit
def vectorized_lower_triangular_cholesky(A:np.ndarray) -> np.ndarray:
    """
        Takes the cholesky decomposition of A vectorized the output
    """
    chex.assert_rank(A, 3)
    chex.assert_equal(A.shape[1], A.shape[2])

    A_chol_flattened = jax.vmap(
        lower_triangular_cholesky,
        [0],
        0
    )(A)

    return A_chol_flattened

@jit
def vectorized_cholesky_to_psd(A:np.ndarray) -> np.ndarray:
    """
    Computes L @ L.T along the first axis
    """
    chex.assert_rank(A, 3)
    chex.assert_equal(A.shape[1], A.shape[2])

    A_psd = jax.vmap(
        lambda L: L @ L.T,
        [0],
        0
    )(A)

    return A_psd

@partial(jit, static_argnums=(1,))
def lower_triangle(val, N):
    tri = np.zeros((N, N))
    #return jax.ops.index_update(tri, jax.ops.index[np.tril_indices(N, 0)], val)
    idx = np.tril_indices(N, 0)
    return tri.at[idx].set(val)



@partial(jit, static_argnums=(1,))
def vectorized_lower_triangular(val:np.ndarray, N) -> np.ndarray:
    return jax.vmap(
        lambda x: lower_triangle(x, N),
        [0],
        0
    )(val)

@jit
def stack_columns(A):
    """
    Stacks columns of A:

                    [0]
        [0, 1]  ->  [2]
        [2, 3]      [1]
                    [3]
    """ 
    chex.assert_rank(A, 2)
    return A.T.reshape(A.shape[0]*A.shape[1], 1)

vec_columns = stack_columns

@jit 
def stack_rows(A):
    """
    Stacks rows of A:

                    [0]
        [0, 1]  ->  [1]
        [2, 3]      [2]
                    [3]
    """ 
    chex.assert_rank(A, 2)
    return np.vstack(A[..., None])

vec_rows = stack_rows

@partial(jit, static_argnums=(1, 2))
def p_get_block_diagonal(A, b_size, A_dim):
    chex.assert_rank(A, 2)

    if b_size == 1:
        return np.diagonal(A)[:, None]

    if b_size == A_dim:
        return A

    raise NotImplementedError()

@partial(jit, static_argnums=(1, 2))
def v_get_block_diagonal(A, b_size, A_dim):
    return jax.vmap(
        p_get_block_diagonal, 
        (0, None, None),
        0
    )(A, b_size, A_dim)

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_block(A, i1:int, i2:int, B1:int, B2:int):
    """
    A is a matrix made up of (evenly sized) B1 x B2 blocks.

    Return the [i1,i2]'th block of A

    Indexing starts at zero.
    """

    chex.assert_rank(A, 2)

    block_size_1 = int(A.shape[0]/B1)
    block_size_2 = int(A.shape[1]/B2)


    return A[
        i1 * block_size_1 : (i1+1) * block_size_1,
        i2 * block_size_2 : (i2+1) * block_size_2,
    ]

@jit
def mat_inv(A):
    A_chol = cholesky(add_jitter(A, settings.jitter))
    return cholesky_solve(A_chol, np.eye(A.shape[0]))

@jit
def solve_with_additive_inverse(A, B_inv, C):
    """ 
    Solves equations of the form (A + B)^{-1} C without forming B
    
    Rewrite as:
        (A + B)^{-1} C = [(A B_inv + I) B ]^{-1} C
                       = B_inv [ A B_inv + I]^{-1} C
    """
    I = np.eye(A.shape[0])
    tmp = A @ B_inv + I
    #tmp = force_symmetric(tmp)

    if True:
        return B_inv @ np.linalg.solve(tmp, C)
    else:
        return B_inv @ cg(tmp, C)[0]

@jit
def force_symmetric(A):
    return 0.5 * (A+A.T)

@partial(jit, static_argnums=(4 ,5))
def lti_disc(F, Q, L, dt, jitter, block_size):
    """ Matrix Fraction Decomposition """
    zeros = np.zeros_like(F)
    eye = np.eye(F.shape[0])
    CD = expm(np.block([[F, L @ Q @ L.T], [zeros, -F.T]])*dt, max_squarings=64) @ np.vstack([zeros, eye])
    C = CD[:block_size]
    D = CD[block_size:]
    Sigma = np.linalg.solve(D.T, C.T).T
    return Sigma

