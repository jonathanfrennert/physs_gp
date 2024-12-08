""" Handling of different parameterisations """
import jax
import jax.numpy as np
from jax import  grad, jit, jacfwd, vjp
from ...dispatch import dispatch, evoke

# Types imports
from ...approximate_posteriors import ConjugateApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullConjugateGaussian, FullGaussianApproximatePosterior, DataLatentBlockDiagonalApproximatePosterior, ApproximatePosterior, DiagonalGaussianApproximatePosterior, MeanFieldConjugateGaussian
from ...sparsity import NoSparsity, FreeSparsity, Sparsity, SpatialSparsity

from .exponential_family_transforms import xi_to_theta, theta_to_lambda, xi_to_expectation, expectation_to_xi, lambda_to_theta, theta_to_xi, theta_to_lambda_diagonal, lambda_to_theta_diagonal, reparametise_cholesky_grad, lambda_to_theta_precision, theta_precision_to_lambda

# =================== Different Parameterisations Entry Points ====================
@dispatch('VGP', MeanFieldConjugateGaussian, "NG_Moment")
def natural_gradients(model, parameterisation, beta: float, enforce_psd_type, prediction_samples: int = None) -> np.ndarray:
    q = model.approximate_posterior
    prior = model.prior
    sparsity_arr = prior.base_prior.get_sparsity_list()

    # compute natural gradient update in natural parameterisation
    # TODO: assuming sparsity is constant across all latents
    lambda_1, lambda_2 = evoke('natural_gradients', model, q, sparsity_arr[0])(
        model, beta, enforce_psd_type, parameterisation
    )
    # convert to mean and covariance
    # vmap over P and blocks

    # Lambda_1 is in Q x Nt x 1 x B
    # Lambda_2 is in Q x Nt x B x B
    theta_1, theta_2 = jax.vmap(jax.vmap(lambda_to_theta))(
        np.reshape(lambda_1, [lambda_1.shape[0], lambda_1.shape[1], -1, 1]), # Q x Nt x B x 1
        lambda_2
    )
    # convert back to the same shame as lambda_1
    theta_1 = np.reshape(theta_1, lambda_1.shape)
    return theta_1, theta_2

@dispatch('VGP', MeanFieldConjugateGaussian, "NG_Precision")
def natural_gradients(model, parameterisation, beta: float, enforce_psd_type, prediction_samples: int = None) -> np.ndarray:
    q = model.approximate_posterior
    prior = model.prior
    sparsity_arr = prior.base_prior.get_sparsity_list()

    # compute natural gradient update in natural parameterisation
    # TODO: assuming sparsity is constant across all latents
    lambda_1, lambda_2 = evoke('natural_gradients', model, q, sparsity_arr[0])(
        model, beta, enforce_psd_type, parameterisation
    )

    # convert to mean and precision
    # vmap over P and blocks
    # Lambda_1 is in Q x Nt x 1 x B
    # Lambda_2 is in Q x Nt x B x B

    theta_1, theta_2 = jax.vmap(jax.vmap(lambda_to_theta_precision))(
        lambda_1[:, :, 0, :][..., None], # Q x Nt x B x 1
        lambda_2
    )
    # convert back to the same shame as lambda_1
    theta_1 = theta_1[:, :, None, :, 0]
    return theta_1, theta_2

@dispatch('VGP', FullConjugateGaussian, "NG_Moment")
def natural_gradients(model, parameterisation, beta: float, enforce_psd_type, prediction_samples: int = None) -> np.ndarray:
    q = model.approximate_posterior
    prior = model.prior
    sparsity_arr = prior.base_prior.get_sparsity_list()

    # compute natural gradient update in natural parameterisation
    # TODO: assuming sparsity is constant across all latents
    lambda_1, lambda_2 = evoke('natural_gradients', model, q, sparsity_arr[0])(
        model, beta, enforce_psd_type, parameterisation
    )

    theta_1, theta_2 = jax.vmap(lambda_to_theta)(
        lambda_1, 
        lambda_2
    )

    if False:
        l1, l2 = jax.vmap(theta_to_lambda)(theta_1, theta_2)
        theta_1, theta_2 = jax.vmap(lambda_to_theta)(l1, l2)
        l1, l2 = jax.vmap(theta_to_lambda)(theta_1, theta_2)
        theta_1, theta_2 = jax.vmap(lambda_to_theta)(l1, l2)

    Y_shape = q.surrogate.data._Y.value.shape
    theta_1 = np.reshape(theta_1, Y_shape)

    print('OUT: ', 'theta_1: ', np.sum(theta_1), 'theta_1: ', np.sum(theta_2))

    #breakpoint()

    return theta_1, theta_2

@dispatch('VGP', FullConjugateGaussian, "NG_Precision")
def natural_gradients(model, parameterisation, beta: float, enforce_psd_type, prediction_samples: int = None) -> np.ndarray:
    q = model.approximate_posterior
    prior = model.prior
    sparsity_arr = prior.base_prior.get_sparsity_list()

    # compute natural gradient update in natural parameterisation
    # TODO: assuming sparsity is constant across all latents
    lambda_1, lambda_2 = evoke('natural_gradients', model, q, sparsity_arr[0])(
        model, beta, enforce_psd_type, parameterisation
    )

    theta_1, theta_2 = jax.vmap(lambda_to_theta_precision)(
        lambda_1, 
        lambda_2
    )

    Y_shape = q.surrogate.data._Y.value.shape
    theta_1 = np.reshape(theta_1, Y_shape)

    return theta_1, theta_2
