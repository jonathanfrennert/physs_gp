""" 
Negative Log Predictive Density (NLPD) computations.

The NLPD is given by:
    NLPD = - log p(Y*_n | X, Y) =  - log ∫ p(Y*_n | F*_n) p(F*_n | X, Y) d F*_n

In the variational settings we approximate p(F*_n | X, Y) with q(F*_n). 

There are three settings:

    Exact computation:
        When the integrals are analytically we compute the NLPD exactly

    Quadrature computation:
        When the integrals are not analytical and the the likelihood decomposes we use quadrature

    Monte-carlo Computation:
        In all other cases we use a monte-carlo estimation.


    In the non-analytical settings we rewrite the NLPD using the Log-Exp trick (for numerical stability):
"""

import jax
import jax.numpy as np
from jax.scipy.special import logsumexp
import chex
from batchjax import batch_or_loop, BatchType

from ..dispatch import dispatch, evoke
from ..computation.gaussian import log_gaussian, log_gaussian_scalar
from ..utils.nan_utils import get_same_shape_mask
from ..utils.utils import get_batch_type

# Import Types
from ..core import Model
from ..data import Data, TransformedData, TemporallyGroupedData
from ..likelihood import Likelihood, GaussianProductLikelihood, ProductLikelihood
from ..transforms import Independent, LinearTransform, Transform, NonLinearTransform, DataLatentPermutation
from ..inference import Inference, Variational, Batch
from ..core.model_types import get_model_type, LinearModel, NonLinearModel
from ..models import VGP

@dispatch(Data, 'VGP', ProductLikelihood, NonLinearModel, 'Variational')
@dispatch(Data, 'VGP', ProductLikelihood, LinearModel, 'Variational')
def nlpd(XS, YS, m, prior, num_samples = None):
    """
    For each n:
        NLPD = - log p(Y*_n | X, Y)
            \approx - log (1\S) \sum^S_s p(Y*_n | F^(s)_n) for F^(s)_n \sim q(F_n)

        and we compute inner sum using the log sum exp trick
        
    """

    mask = get_same_shape_mask(YS)
    Y_masked = np.nan_to_num(YS, nan=0.0)

    #compute samples
    # shape will be [n_samples, N, P, 1] or [n_samples, N, P]

    mu = evoke('marginal_prediction_samples', m.approximate_posterior, m.likelihood, m.prior, whiten=m.inference.whiten)(
        XS, m.data, m.approximate_posterior, m.likelihood, m.prior, m.inference, 1, m.inference.whiten, num_samples = num_samples
    )

    P = YS.shape[1]
    N = XS.shape[0]
    n_samples = mu.shape[0]

    # ensure consistent shape
    mu = np.reshape(mu, [n_samples, N, P])

    lik_arr = m.likelihood.likelihood_arr

    ll_arr = batch_or_loop(
        lambda f_ns, y_p, lik: jax.vmap(lambda f, y: lik.log_likelihood(f[:, None], y[:, None]), [0, None])(f_ns, y_p),
        [mu, Y_masked, lik_arr],
        [2, 1, 0],
        dim = len(lik_arr),
        out_dim=1,
        batch_type = get_batch_type(lik_arr)
    )
    chex.assert_shape(ll_arr, [P, n_samples, N])
    # ll_arr will have shape [P, n_samples, N]
    # average over n_samples for each N
    ll_arr = ll_arr

    # vmap over P
    res = jax.vmap(
        lambda ll: jax.vmap(lambda l: logsumexp(l), [1])(ll), #vmap over N
        [0]
    )(ll_arr)

    res = res + np.log(1/n_samples)

    # vmap of N, sum of P and logsumexp over n_samples
    res_global = jax.vmap(
        lambda l: logsumexp(np.sum(l, axis=0)),  #sum of N
        [2]
    )(ll_arr)
    res_global = res_global + np.log(1/n_samples)
    res_global = res_global[..., None]

    chex.assert_shape(res, [P, N])
    chex.assert_shape(res_global, [N, 1])

    # N x P
    res = res.T

    # mask res
    res = res * mask

    # mask joint res
    mask_global = np.prod(mask, axis=1)[:, None]
    res_global = res_global * mask_global

    # Average over N, ignoring the missing data
    return - np.sum(res, axis=0) / np.sum(mask, axis=0), - np.sum(res_global, axis=0) / np.sum(mask_global, axis=0)

@dispatch(Data, Model, GaussianProductLikelihood, LinearModel, Inference)
@dispatch(TemporallyGroupedData, Model, GaussianProductLikelihood, LinearModel, Inference)
@dispatch(Data, 'VGP', GaussianProductLikelihood, LinearModel, 'Variational')
@dispatch(TemporallyGroupedData, 'VGP', GaussianProductLikelihood, LinearModel, 'Variational')
def nlpd(XS, YS, m, prior, num_samples=None):
    """ Closed form Gaussian NLPD """

    N, P = YS.shape

    if False:
        pred_mu, pred_var = m.predict_y(XS, diagonal=False, squeeze=False)
        pred_var_diag = np.diagonal(pred_var, axis1=1, axis2=2)
    else:
        pred_mu, pred_var_diag = m.predict_y(XS, diagonal=True, squeeze=False)
        if len(pred_var_diag.shape) == 2:
            pred_var = jax.vmap(np.diag)(pred_var_diag)
        elif len(pred_var_diag.shape) == 4:
            pred_var = pred_var_diag[..., 0]
            pred_var_diag = pred_var_diag[..., 0, 0]
            pred_mu = pred_mu[..., 0]

    #compute NLPD independtly for each likelihood

    chex.assert_rank([YS, pred_mu, pred_var_diag], [2, 2, 2])
    chex.assert_equal_shape([YS, pred_mu, pred_var_diag])

    mask = get_same_shape_mask(YS)
    Y_masked = np.nan_to_num(YS, nan=0.0)

    res = jax.vmap(
        jax.vmap(log_gaussian_scalar, [0, 0, 0]),
        [0, 0, 0]
    )(Y_masked, pred_mu, pred_var_diag)

    # mask res
    res = res * mask

    # compute joint NLPD
    chex.assert_rank([YS, pred_mu, pred_var], [2, 2, 3])
    chex.assert_equal_shape([YS, pred_mu])
    chex.assert_shape(pred_var, [N, P, P])

    mask = get_same_shape_mask(YS)
    Y_masked = np.nan_to_num(YS, nan=0.0)

    # TODO: compute the point-wise and the GLOBAL ones
    res_global = jax.vmap(
        log_gaussian,
        [0, 0, 0]
    )(Y_masked[..., None], pred_mu[..., None], pred_var) # ensure rank 2 after batching
    res_global = res_global[..., None]

    mask_global = np.prod(mask, axis=1)[:, None]

    chex.assert_equal_shape([res_global, mask_global])

    # mask res
    res_global = res_global * mask_global

    # Average over N, ignoring the missing data
    return - np.sum(res, axis=0) / np.sum(mask, axis=0), - np.sum(res_global, axis=0) / np.sum(mask_global, axis=0)


@dispatch(TransformedData, Model, Likelihood, LinearModel, Inference)
@dispatch(TransformedData, Model, Likelihood, NonLinearModel, Inference)
def nlpd(XS, YS, model, prior):
    """
    In the transformed setting the NLPD is given as:

         - (1/N) \sum^N_n [ \log p(T(YS_n)) + log |dT(YS_n) / d YS_2| ]
    """

    model_type = get_model_type(model.prior)

    data = model.data
    base_data = data.base_data

    base_nlpd = evoke('nlpd', base_data, model, model.likelihood, model.prior, model.inference)(
        XS,
        data.forward_transform(YS),
        model,
        model.prior
    )
    
    mask = get_same_shape_mask(YS)

    log_jac = data.log_jacobian(YS)
    log_jac = np.nan_to_num(log_jac, 0.0)

    # Average over N, ignoring the missing data
    log_jac = np.sum(log_jac, axis=0) / np.sum(mask, axis=0)

    return base_nlpd - log_jac


# =========================== Batch Models  ===========================
@dispatch('BatchGP')
def nlpd(XS, YS, model, num_samples = None):
    # samples not required for these models

    model_type = get_model_type(model.prior)

    return evoke('nlpd', model.data, model, model.likelihood, model_type, model.inference)(
        XS,
        YS,
        model,
        model.prior
    )

# =========================== Variational Models  ===========================
@dispatch('VGP')
def nlpd(XS, YS, model, num_samples = None):
    """ 
    In the variational setting the NLPD is approximated as:
    NLPD = - log p(Y*_n | X, Y) =  - log ∫ p(Y*_n | F*_n) q(F*_n) d F*_n
    """

    model_type = get_model_type(model.prior)


    return evoke('nlpd', model.data, model, model.likelihood, model_type, model.inference)(
        XS,
        YS,
        model,
        model.prior,
        num_samples = num_samples
    )

# =========================== Entry Point  ===========================
def nlpd(XS, YS, model, num_samples = None):
    if model.data.minibatch:
        model.data.batch()

    return evoke('nlpd', model)( XS, YS, model, num_samples=num_samples)

