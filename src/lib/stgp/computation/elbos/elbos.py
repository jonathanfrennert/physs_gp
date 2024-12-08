import jax
import jax.numpy as np
from jax import jit
import objax
import chex

from batchjax import batch_or_loop
from ...utils.utils import get_batch_type
from ... import settings
from ...approximate_posteriors import ApproximatePosterior, ConjugateApproximatePosterior, FullConjugateGaussian, MeanFieldConjugateGaussian
from ...transforms import Independent, Transform
from ...likelihood import Likelihood
from ...dispatch import dispatch, evoke, _ensure_str

from ...utils.nan_utils import get_same_shape_mask

from .prior_ops import prior_mean_Z, prior_covar_ZZ, prior_covar_XZ

def compute_expected_log_liklihood_with_variational_params(data, q_m, q_S, likelihood, prior, approximate_posterior, inference):

    N = data.N

    if True:
        if data.minibatch:
            # TODO: minibatching only works when sparsity is used. Assert this.
            data.batch()

    q_f_mu, q_f_var = evoke('marginal', approximate_posterior, likelihood, prior, whiten=inference.whiten)(
        data, q_m, q_S, approximate_posterior, likelihood, prior, inference.whiten
    )


    # Compute Expected Log Likelihood   
    ELL = evoke('expected_log_likelihood', data, likelihood, prior, approximate_posterior)(
        data, q_f_mu, q_f_var, likelihood, prior, approximate_posterior, inference
    )

    if data.minibatch:
        #chex.assert_shape(ELL, [prior.output_dim])
        Y_mask = get_same_shape_mask(data.Y)
        return data.minibatch_scaling * np.sum(ELL)
    else:
        return np.sum(ELL)

def compute_expected_log_liklihood(data, likelihood, prior, approximate_posterior, inference):

    # We use the base prior because we want to get the latent prior, not the transformed ones
    q_m, q_S = evoke('variational_params', approximate_posterior, likelihood, prior.base_prior, inference.whiten)(
        data, approximate_posterior, likelihood, prior, inference.whiten
    )

    return compute_expected_log_liklihood_with_variational_params(
        data, q_m, q_S, likelihood, prior, approximate_posterior, inference
    )

@dispatch(Likelihood, Transform, ApproximatePosterior)
def elbo(
    data, likelihood: Likelihood, prior: Independent, approximate_posterior: ApproximatePosterior, inference: 'Variational'
):
    # Compute KL term
    KL = evoke('kullback_leibler', approximate_posterior, prior, whiten=inference.whiten)(
        approximate_posterior, prior, inference.whiten
    )

    # Compute expected log likelihood term
    ELL = compute_expected_log_liklihood(data, likelihood, prior, approximate_posterior, inference)

    print(f'ELL: {ELL}, KL: {KL}')

    return  ELL - KL

# ============ CVI ELBOs ==================

@dispatch(Likelihood, Transform, ConjugateApproximatePosterior)
@dispatch(Likelihood, Transform, FullConjugateGaussian)
def cvi_kl(
    data, likelihood: Likelihood, prior: Transform, approx_posterior: ConjugateApproximatePosterior, inference: 'Variational'
):
   # Compute surrogate ELL
    ELL_surrogate = compute_expected_log_liklihood(
        approx_posterior.surrogate.data, 
        approx_posterior.surrogate.likelihood, 
        prior.base_prior, 
        approx_posterior, 
        inference
    )
    ML_surrogate = - approx_posterior.surrogate.get_objective()

    negKL =  - ELL_surrogate + ML_surrogate
    return -negKL


#@dispatch(Likelihood, Transform, ConjugateApproximatePosterior)
def elbo(
    data, likelihood: Likelihood, prior: Transform, approx_posterior: ConjugateApproximatePosterior, inference: 'Variational'
):
    # Compute ELL
    ELL = compute_expected_log_liklihood(data, likelihood, prior, approx_posterior, inference)

    # Compute surrogate ELL
    # TODO: this needs to be generalised to map across all X

    surrogate_ell_fn = lambda q, q_p, inf: compute_expected_log_liklihood(q.surrogate.data, q.surrogate.likelihood.likelihood_arr[0], q_p, q, inf)
    #surrogate_ell_fn = lambda q, q_p, inf: compute_expected_log_liklihood(q.surrogate.data, q.surrogate.likelihood, q_p, q, inf)

    q_list = approx_posterior.approx_posteriors

    # TODO: assume q(u) is an independent
    ELL_surrogate =  batch_or_loop(
        surrogate_ell_fn,
        [q_list, prior.base_prior.parent, inference],
        [0, 0, None],
        dim=len(q_list),
        out_dim=1,
        batch_type = get_batch_type(q_list)
    )

    ELL_surrogate = np.sum(ELL_surrogate)

    # get_ojective returns the negative log liklihood
    # We require the (postive) log liklihood
    ML_arr =  batch_or_loop(
        lambda qq: -qq.surrogate.get_objective(),
        [q_list],
        [0],
        dim=len(q_list),
        out_dim = 1,
        batch_type = get_batch_type(q_list)
    )

    ML_surrogate = np.sum(ML_arr)


    return ELL - ELL_surrogate + ML_surrogate

@dispatch(Likelihood, Transform, MeanFieldConjugateGaussian)
def elbo(
    data, likelihood: Likelihood, prior: Transform, approx_posterior: ConjugateApproximatePosterior, inference: 'Variational'
):
    # Compute ELL
    ELL = compute_expected_log_liklihood(data, likelihood, prior, approx_posterior, inference)

    # compute KL
    q_list = approx_posterior.approx_posteriors
    kl_fn = evoke('cvi_kl', likelihood, prior, q_list[0])

    latents_arr = prior.base_prior.parent

    KL_arr =  batch_or_loop(
        lambda qq, ll: kl_fn(data, likelihood, ll, qq, inference),
        [q_list, latents_arr],
        [0, 0],
        dim=len(q_list),
        out_dim = 1,
        batch_type = get_batch_type(q_list)
    )
    KL = np.sum(KL_arr)


    return ELL - KL


@dispatch(Likelihood, Transform, FullConjugateGaussian)
def elbo(
    data, likelihood: Likelihood, prior: Transform, q: ConjugateApproximatePosterior, inference: 'Variational'
):
    # these will be in time-latent-space format
    # when calling compute_expected_log_liklihood_with_variational_params this will call a marginal that will convert it to time-space-latent format
    lml, q_m, q_S =  q.surrogate.posterior_blocks(return_lml=True)

    if settings.verbose:
        print(f'lml: {np.sum(lml)}, q_m: {np.sum(q_m)}, q_S: {np.sum(q_S)}')

    # data is stored in time-space-latent format
    ELL =  compute_expected_log_liklihood_with_variational_params(
        data, q_m, q_S, likelihood, prior, q, inference
    )

    ELL_surrogate = compute_expected_log_liklihood_with_variational_params(
        q.surrogate.data, 
        q_m, q_S,
        q.surrogate.likelihood, 
        prior.base_prior, 
        q, 
        inference
    )
    ML_surrogate = lml

    elbo =  ELL - ELL_surrogate + ML_surrogate

    if settings.verbose:
        print(f'ELL: {ELL}, ELL_surrogate: {ELL_surrogate}, ML_surrogate: {ML_surrogate}, KL: {-ELL_surrogate + ML_surrogate}')

    return elbo
