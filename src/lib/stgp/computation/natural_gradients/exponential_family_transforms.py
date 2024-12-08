from ... import settings
from ..matrix_ops import cholesky, cholesky_solve, triangular_solve, add_jitter, lower_triangle, vectorized_lower_triangular_cholesky, vectorized_lower_triangular, lower_triangular_cholesky, triangular_solve

import jax
import jax.numpy as np
from jax.scipy.sparse.linalg import cg
from jax import  jit, vjp
import chex
from functools import partial

@jit
def xi_to_theta(xi1, xi2):
    return xi1, xi2 @ xi2.T

@jit
def theta_to_xi(theta_1, theta_2):
    M = theta_1.shape[0]
    jit = settings.ng_jitter * np.eye(M) 

    xi1 = theta_1
    xi2 = cholesky(theta_2+jit)

    return xi1, xi2

@jit
def theta_to_lambda(theta_1, theta_2):
    M = theta_1.shape[0]
    jit = settings.ng_jitter * np.eye(M) 

    theta_2_chol = cholesky(theta_2+jit)
    I = np.eye(theta_2_chol.shape[0])


    if True:
        lambda_1 = cholesky_solve(theta_2_chol, theta_1)
        lambda_2 = -0.5*cholesky_solve(theta_2_chol, I)
        #lambda_2 = -0.5*theta_2_chol_inv.T @ theta_2_chol_inv
    else:
        lambda_1 = cg(theta_2, theta_1)[0]
        lambda_2 = -0.5*cg(theta_2, I)[0]

    return lambda_1, lambda_2

@jit
def theta_precision_to_lambda(theta_1, theta_2):
    M = theta_1.shape[0]
    jit = settings.ng_jitter * np.eye(M) 

    theta_2_chol = cholesky(theta_2+jit)
    lambda_1 = cholesky_solve(theta_2_chol, theta_1)
    lambda_2 = -0.5*theta_2

    return lambda_1, lambda_2


@jit
def theta_to_lambda_diagonal(theta_1, theta_2):
    lambda_1 = theta_1 / theta_2
    lambda_2 = -0.5/theta_2

    return lambda_1, lambda_2

@jit
def lambda_to_theta_diagonal(lambda_1, lambda_2):
    theta_2 = 1/(-2*lambda_2)
    theta_1 = theta_2 * lambda_1

    return theta_1, theta_2

@jit
def lambda_to_theta(lambda_1, lambda_2):
    M = lambda_1.shape[0]

    lambda_2_chol = cholesky(add_jitter(-2*lambda_2, settings.ng_jitter))

    theta_2 =  cholesky_solve(lambda_2_chol, np.eye(M))
    #theta_2_sqrt =  triangular_solve(lambda_2_chol, np.eye(M))
    #theta_2 = theta_2_sqrt.T @ theta_2_sqrt
    theta_1 =  theta_2 @ lambda_1

    #theta_1 =  cholesky_solve(lambda_2_chol, lambda_1)

    return theta_1, theta_2

@jit
def lambda_to_theta_precision(lambda_1, lambda_2):
    M = lambda_1.shape[0]
    jit = settings.ng_jitter * np.eye(M) 

    lambda_2_chol = cholesky(-2*lambda_2+jit)

    theta_2 =  -2*lambda_2
    theta_1 =  cholesky_solve(lambda_2_chol, lambda_1)

    return theta_1, theta_2

@jit
def xi_to_lambda(xi1, xi2):
    theta_1, theta_2 = xi_to_theta(xi1, xi2)
    return theta_to_lambda(theta_1, theta_2)

@jit
def lambda_to_xi(lambda_1, lambda_2):
    theta_1, theta_2 = lambda_to_theta(lambda_1, lambda_2)
    return theta_to_xi(theta_1, theta_2)

@jit
def xi_to_expectation(xi1, xi2):
    xi2 = xi2 @ xi2.T
    return xi1, xi1 @ xi1.T + xi2

@jit
def expectation_to_xi(mu1, mu2):
    xi2 = cholesky(mu2 - mu1 @ mu1.T)
    return mu1, xi2



@partial(jit, static_argnums=(3))
def reparametise_cholesky_grad(s, s_chol_grad, prior, permute_flag):
    #Calculate ∂L/μ = ∂L/∂ξ ∂ξ/μ 

    if prior is None:
        x, u = vjp(
            lambda A: cholesky(add_jitter(A, settings.ng_jitter)), 
            s
        )
    else:
        x, u = vjp(
            lambda A: cholesky(add_jitter(prior.unpermute_mat(A), settings.ng_jitter)), 
            s
        )

    s_grad = u(s_chol_grad)[0]

    # Symmetrize gradient (in case jax.scipy.linalg.cholesky is used)
    s_grad = s_grad/2 
    s_grad = s_grad + s_grad.T

    return s_grad

