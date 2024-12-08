""" Closed form y predictions of mean and variance of p(y | XS) """
import jax
from jax import jit
import jax.numpy as np
import chex
import warnings

from ...likelihood import ProductLikelihood, Gaussian
from ...transforms import LinearTransform, Independent, Transform
from ...dispatch import dispatch, evoke, DispatchNotFound, _ensure_str
from ...utils.batch_utils import batch_over_module_types
from batchjax import batch_or_loop, BatchType
from ..matrix_ops import add_jitter, vec_add_jitter
from ..model_ops import get_vec_gaussian_likelihood_variances
from ...core import Posterior
from ... import settings

# Gaussian Likelihoods

# ======= Full var ========
@dispatch(Posterior, 'Gaussian')
def predict_y_full(XS, likelihood, post_mu, post_var):
    chex.assert_rank([post_mu, post_var], [2, 3])
    breakpoint()
    return post_mu, add_jitter(post_var, likelihood.variance)

@dispatch(Posterior, 'GaussianProductLikelihood')
def predict_y_full(XS, likelihood, post_mu, post_var):
    # assert all likelihoods are gaussian
    chex.assert_rank([post_mu, post_var], [2, 3])
    N, P = post_mu.shape

    lik_var = np.diag(likelihood.variance)
    chex.assert_shape(lik_var, [P, P])

    return post_mu, vec_add_jitter(post_var, lik_var)

# ======= Diagonal var ========

@dispatch(Posterior, 'NonZeroLoss')
def predict_y_diagonal(XS, likelihood, post_mu, post_var):
    chex.assert_rank([post_mu, post_var], [2, 3])
    chex.assert_equal([post_var.shape[1], post_var.shape[2]], [1, 1])

    return likelihood.conditional_mean(post_mu), post_var


@dispatch(Posterior, 'Gaussian')
def predict_y_diagonal(XS, likelihood, post_mu, post_var):
    chex.assert_rank([post_mu, post_var], [2, 3])
    chex.assert_equal([post_var.shape[1], post_var.shape[2]], [1, 1])

    return post_mu, post_var + likelihood.variance

@dispatch(Posterior, 'ProductLikelihood')
def predict_y_diagonal(XS, likelihood, post_mu, post_var):
    assert len(likelihood.likelihood_arr) == 1
    return  evoke('predict_y_diagonal', Posterior, likelihood.likelihood_arr[0])(
        XS, likelihood.likelihood_arr[0], post_mu, post_var
    )


@dispatch(Posterior, 'ReshapedGaussian')
def predict_y_diagonal(XS, likelihood, post_mu, post_var):
    chex.assert_rank([post_mu, post_var], [3, 4])
    chex.assert_equal([post_var.shape[2], post_var.shape[3]], [1, 1])

    # TODO: small hack to add support for DiagionalGaussian
    if _ensure_str(likelihood.base)=='DiagonalGaussian':
        # extract the diagonal of the DiagonalGaussian first
        # then tile/repeat for each of the prediction locations
        lik_var = np.tile(np.diag(likelihood.base.variance), [post_var.shape[0], 1])[..., None, None]
    else:
        # make the likelihood variance the same shape as the (predictive) posterior variance
        lik_var = np.tile(likelihood.base.variance, [post_var.shape[0], 1])[..., None, None]

    chex.assert_equal([lik_var.shape], [post_var.shape])

    return post_mu, post_var + lik_var

@dispatch('FullConjugateGaussian', 'HetGaussian')
def predict_y_diagonal(XS, likelihood, post_mu, post_var):
    warnings.warn('post_var should be full matrix -- approximating with diagonal for now')
    post_mu = np.squeeze(post_mu)
    post_var = np.squeeze(post_var)

    m_f = post_mu[:, 0][:, None]
    m_g = post_mu[:, 1][:, None]

    k_f = post_var[:, 0][:, None]
    k_g = post_var[:, 1][:, None]

    mean = m_f
    var = k_f + np.exp(2 * m_g + 2 * k_g)

    var = k_f + m_g**2

    return mean, var[..., None]

# ======= Dispatchers ========

@dispatch(Posterior, "HetGaussian", Transform)
def predict_y(XS, gp, likelihood, post_mu, post_var, diagonal: bool):
    if diagonal:
        m_f = post_mu[:, 0, ...][:, None, ...]
        m_g = post_mu[:, 1, ...][:, None, ...]

        k_f = post_var[:, 0, ...]
        k_g = post_var[:, 1, ...]

        mean = m_f
        var = k_f + np.exp(2 * m_g + 2 * k_g)

        return mean, m_g**2

@dispatch(Posterior, ProductLikelihood, Transform)
def predict_y(XS, gp, likelihood, post_mu, post_var, diagonal: bool):

    if type(post_mu) == list:
        post_mu = np.array(post_mu)
        post_var = np.array(post_var)
        if settings.experimental_allow_f_multi_dim_per_output:
            post_mu = np.transpose(post_mu, [1, 0, 2, 3])
            post_var = np.transpose(post_var, [1, 0, 2, 3, 4])
            chex.assert_rank([post_mu, post_var], [4, 5])
        else:

            post_mu = np.transpose(post_mu, [1, 0, 2, 3])[:, :, 0, :]
            post_var = np.transpose(post_var, [1, 0, 2, 3, 4])[:, :, 0, :, :]

            chex.assert_rank([post_mu, post_var], [3, 4])
    else:
        chex.assert_rank([post_mu, post_var], [3, 4])

    if diagonal:
        likelihood_arr = likelihood.likelihood_arr
        num_outputs = len(likelihood_arr)

        lik_types = [type(lik)==Gaussian for lik in likelihood_arr]
        if all(lik_types):

            # Compute prediction for each likelihood-prior pair
            mu_arr, var_arr =  batch_over_module_types(
                'predict_y_diagonal',
                [gp],
                likelihood_arr,
                [XS, likelihood_arr, post_mu, post_var],
                [None, 0, 1, 1],
                num_outputs,
                2
            )
        else:
            # only compute closed form  ones
            mu_arr, var_arr = [], []

            for p in range(num_outputs):
                try:
                    mu_p, var_p = evoke('predict_y_diagonal', gp, likelihood_arr[p])(
                        XS, likelihood_arr[p], post_mu[:, p], post_var[:, p]
                    )
                    mu_arr.append(mu_p)
                    var_arr.append(var_p)
                except DispatchNotFound as e:
                    print(e)
                    print('Likelihood dispatch not found!')
                    mu_arr.append(post_mu[:, p]*np.nan)
                    var_arr.append(post_var[:, p]*np.nan)

            mu_arr = np.array(mu_arr)
            var_arr = np.array(var_arr)

        # fix data-latent ordering due to batching
        mu_arr = np.transpose(mu_arr, [1, 0, 2])
        var_arr = np.transpose(var_arr, [1, 0, 2, 3])

        return mu_arr, var_arr

    else:
        # TODO: this will break with aggreagtion
        # fix shapes
        post_mu = post_mu[..., 0]
        post_var = post_var[:, 0, ...]

        mu, var =  evoke('predict_y_full', gp, likelihood)(
            XS, likelihood, post_mu, post_var
        )

        # add back removed dimensions
        mu = mu[..., None]
        var = var[:, None, ...]

        chex.assert_rank([mu, var], [3, 4])

        return mu, var

@dispatch('BatchGP', 'ReshapedGaussian', LinearTransform)
def predict_y(XS, gp, likelihood, post_mu, post_var, diagonal: bool):
    chex.assert_rank([post_mu, post_var], [3, 4])

    if diagonal:
        lik_var = get_vec_gaussian_likelihood_variances(
            np.transpose(post_var, [1, 0]),
            likelihood_arr
        )

        return post_mu, post_var+lik_var
    else:
        raise NotImplementedError()


@dispatch('BatchGP', 'GaussianProductLikelihood', LinearTransform)
def predict_y(XS, gp, likelihood, post_mu, post_var, diagonal: bool):
    chex.assert_rank([post_mu, post_var], [3, 4])

    if diagonal:
        chex.assert_rank([post_mu, post_var], [2, 2])
        chex.assert_equal_shape([post_mu, post_var])

        lik_var = likelihood.base.variance

        chex.assert_rank(lik_var, 1)

        return post_mu, post_var + lik_var[:, None]
    else:
        raise NotImplementedError()

