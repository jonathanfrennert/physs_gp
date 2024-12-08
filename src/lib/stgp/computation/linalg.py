""" Source of linear solvers """

import jax
import jax.numpy as np
from jax.scipy.sparse.linalg import cg
from jax import jit

from .. import settings
from .matrix_ops import add_jitter, cholesky_solve, cholesky, log_chol_matrix_det
import chex

@jit
def solve(A, B):
    """
    Solves equations of the form Ax=B either exactly or approximately depending on settings.linear_solver

    We support three types of solve:
        cholesky:
            first the cholesky decomposition of A is taken and a cholesky_solve is constructed

        cg:
            approximately solve the equation using conjugate gradients

        exact: 
            solve exactly without exploiting any structure of A or B
    """
    chex.assert_rank([A, B], [2, 2])

    # always add jitter
    A = add_jitter(A, settings.jitter)

    if settings.linear_solver == settings.SolveType.CHOLESKY:
        return solve_from_cholesky(cholesky(A), B)

    elif settings.linear_solver == settings.SolveType.CG:
        # TODO: add a preconditioner?
        return cg(
            A,
            B,
            maxiter = min(A.shape[0], settings.cg_max_iter)
            #maxiter = 5
        )[0]

    elif settings.linear_solver == settings.SolveType.EXACT:
        return np.linalg.solve(A, B)

    raise RuntimeError()

@jit
def solve_from_cholesky(A_chol, B):
    """
    Solves equations of the form Ax=B either exactly or approximately depending on settings.linear_solver
        given A_chol where A = A_chol @ A_chol.T

    If linear_solve is not cholesky then this just forms the full A and calls solve.
    """
    chex.assert_rank([A_chol, B], [2, 2])

    if settings.linear_solver == settings.SolveType.CHOLESKY:
        return cholesky_solve(A_chol, B)

    # if we are not using a cholesky solve then we do not expoloit the cholesky factor
    # so just use the standard solve function with the full A
    A = A_chol @ A_chol.T

    return solve(A, B)

@jit
def log_determinant(A):
    chex.assert_rank([A], [2])

    # always add jitter
    A = add_jitter(A, settings.jitter)

    if settings.linear_solver == settings.SolveType.CHOLESKY:
        return log_determinant_from_cholesky(cholesky(A))

    elif settings.linear_solver == settings.SolveType.CG:
        # compute eigen values
        eigval, eigvec = np.linalg.eigh(A)
        return np.sum(np.log(eigval))

    elif settings.linear_solver == settings.SolveType.EXACT:
        # slogdet returns (sign, logdet), we dont need the sign info so just ignore it
        return np.linalg.slogdet(A)[1]

    raise RuntimeError()

@jit
def log_determinant_from_cholesky(A_chol):
    chex.assert_rank([A_chol], [2])

    if settings.linear_solver == settings.SolveType.CHOLESKY:
        return log_chol_matrix_det(A_chol)

    A = A_chol @ A_chol.T
    return log_determinant(A)

