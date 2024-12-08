import jax
import jax.numpy as np
import chex

from ...dispatch import evoke, dispatch

# Import types
from ...core import Model
from ...transforms import Transform, LinearTransform, NonLinearTransform, DataLatentPermutation, Independent
from ...likelihood import ProductLikelihood, Likelihood
from ...data import Data, TransformedData
from ..integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ...core.model_types import get_model_type, LinearModel, NonLinearModel

@dispatch(Data, Model, Likelihood, NonLinearModel)
def confidence_intervals(XS, m, num_samples = None, y_samples=None):
    # TODO: this is assuming a variational model
    out_block_dim = 1

    if num_samples is None:
        num_samples = m.inference.prediction_samples

    if y_samples is None:
        y_samples = num_samples

    mu = evoke('marginal_prediction_samples', m.approximate_posterior, m.likelihood, m.prior, whiten=m.inference.whiten)(
        XS, m.data, m.approximate_posterior, m.likelihood, m.prior, m.inference, out_block_dim, m.inference.whiten, num_samples = num_samples
    )

    # Ensure correct shape
    mu = np.reshape(mu, [num_samples, XS.shape[0], m.prior.output_dim])

    # for each sample, sample from the likelihood
    lik_samples_fn = lambda f: m.likelihood.conditional_samples(f, generator=m.inference.generator, num_samples=y_samples)

    mu_lik = jax.vmap(lik_samples_fn, [0])(mu)
    # fix shapes
    mu_lik = mu_lik[:, :, :, :, 0 , 0]
    mu_lik = np.transpose(mu_lik, [0, 2, 3, 1])

    mu_lik = np.vstack(mu_lik)
    chex.assert_rank(mu_lik, 3)

    mu = mu_lik

    ci_lower, median, ci_upper  = np.percentile(mu, np.array([2.5, 50, 97.5]), axis=0)

    chex.assert_rank([ci_lower, median, ci_upper], [2, 2, 2])

    # ensure shape is [P, N]
    return median.T, ci_lower.T, ci_upper.T

@dispatch(Data, Model, Likelihood, LinearModel)
def confidence_intervals(XS, m, num_samples = None):
    mu, var = m.predict_y(XS, squeeze=False, diagonal=True)

    P = mu.shape[0]

    # Ensure rank 2
    mu = np.reshape(mu, [P, -1])
    var = np.reshape(var, [P, -1])

    return mu, mu-1.96*np.sqrt(var), mu+1.96*np.sqrt(var)

@dispatch(TransformedData, Model, Likelihood, LinearModel)
@dispatch(TransformedData, Model, Likelihood, NonLinearModel)
def confidence_intervals(XS, m, num_samples = None, **kwargs):
    base_data = m.data.base_data

    model_type = get_model_type(m.prior)

    median, lower_ci, upper_ci = evoke(
        'confidence_intervals', base_data, m, m.likelihood, model_type
    )(XS, m, num_samples = num_samples, **kwargs)

    chex.assert_rank([median, lower_ci, upper_ci], [2, 2, 2])

    median = m.data.inverse_transform(median)
    lower_ci = m.data.inverse_transform(lower_ci)
    upper_ci = m.data.inverse_transform(upper_ci)

    return median, lower_ci, upper_ci

# =========================== Entry Point  ===========================
@dispatch(Model)
def confidence_intervals(XS, m, num_samples=None, **kwargs):
    if m.data.minibatch:
        # TODO: minibatching only works when sparsity is used. Assert this.
        m.data.batch()

    model_type = get_model_type(m.prior)

    return evoke(
        'confidence_intervals', m.data, m, m.likelihood, model_type
    )(XS, m, num_samples = num_samples, **kwargs)
