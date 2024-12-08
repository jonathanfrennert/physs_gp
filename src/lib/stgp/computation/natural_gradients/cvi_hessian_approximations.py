"""
When computing natural gradients we need to compute dE[log P(Y|T(F))]/dS, which is not gurarenteed to ensure P.S.D updates.

Here we compute Gauss-Newton style approximations of this hessian.
"""
import jax
import jax.numpy as np
from jax import  grad, jit, jacfwd, vjp
import chex
import objax
from batchjax import batch_or_loop, BatchType
from functools import partial

from ... import settings
from ...utils.nan_utils import get_same_shape_mask 
from ..matrix_ops import cholesky, cholesky_solve, triangular_solve, vec_add_jitter, add_jitter, lower_triangle, vectorized_lower_triangular_cholesky, vectorized_lower_triangular, lower_triangular_cholesky, lower_triangle, to_block_diag
from ...utils.utils import vc_keep_vars, get_parameters, get_var_name_with_id, get_batch_type
from ..elbos.elbos import compute_expected_log_liklihood, compute_expected_log_liklihood_with_variational_params
from ...dispatch import dispatch, evoke
from ..parameter_transforms import psd_retraction_map
from ..integrals.samples import _process_samples
from ..integrals.approximators import mv_block_monte_carlo, mv_mean_field_block_monte_carlo, mv_block_monte_carlo_list
from ...data import Data, TemporallyGroupedData, MultiOutputTemporalData, TemporalData, SpatioTemporalData, SpatialTemporalInput

from ...dispatch import _ensure_str

# Types imports
from ...approximate_posteriors import ConjugateApproximatePosterior, MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullConjugateGaussian, FullGaussianApproximatePosterior, DataLatentBlockDiagonalApproximatePosterior, ApproximatePosterior, DiagonalGaussianApproximatePosterior, MeanFieldConjugateGaussian
from ...sparsity import NoSparsity, FreeSparsity, Sparsity, SpatialSparsity

from .exponential_family_transforms import xi_to_theta, theta_to_lambda, xi_to_expectation, expectation_to_xi, lambda_to_theta, theta_to_xi, theta_to_lambda_diagonal, lambda_to_theta_diagonal, reparametise_cholesky_grad

from ...transforms import MultiOutput

def get_Y_in_correct_shape(data):
    """
    Return Y in [Nt, Ns, P] shape 
    
    This is necessary because TemporallyGroupedData return Y in a different shape to all other sequential datasets
    """
    Y_st = data.Y_st
    if _ensure_str(data) != 'TemporallyGroupedData':
        Y_st = np.transpose(Y_st, [0, 2, 1])

    return Y_st


def data_decomposes_across_time(data) -> bool:
    time_data_types = ['TemporallyGroupedData', 'SpatioTemporalData', 'TemporalData', 'MultiOutputTemporalData']

    data_type = _ensure_str(data)

    if settings.cvi_ng_exploit_space_time:
        return data_type in time_data_types
    else:
        return False

def create_new_single_time_data_of_same_type(data, X, Y):
    """
    Helper function that creates a new data object using X and Y (which typically will be a single time point)
        and returns an data object with the same type as data.

    This is useful when trying to vmap over the variational approxiamte posterior marginals.
    """

    data_type = _ensure_str(data)
    """ Constructs new data of the same type with new data points X, Y. Does not sort. """
    if data_type == 'Data':
        return Data(X, Y)
    elif data_type == 'TemporallyGroupedData':
        return TemporallyGroupedData(X=X, Y=Y, sort=False)
    elif data_type == 'SpatioTemporalData':
        return SpatioTemporalData(
            X=SpatialTemporalInput(X_time=X[0, 0, :1], X_space = X[0, :, 1:], train=False), 
            Y=Y, 
            sort=False
        )
    elif data_type == 'MultiOutputTemporalData':
        return MultiOutputTemporalData(X=X[0], Y=Y, sort=False)
    elif data_type == 'TemporalData':
        return TemporalData(X=X[0], Y=Y, sort=False)

    raise RuntimeError(f'Data type {data_type} not supported')

def get_likelihood_hessian(model, T_f, laplace_log_lik=False):
    """
        Computes d^2 p(Y | F) / dF df^T either exactly or using a laplace approximation
    """
    data = model.data

    if data_decomposes_across_time(data) and len(T_f.shape) == 4:
        # T_f: Nt x Ns x P x 1
        chex.assert_rank(T_f, 4)

        if not laplace_log_lik:
            Y_st = get_Y_in_correct_shape(model.data)

            hess = batch_or_loop(
                lambda y, t, lik: jax.vmap(jax.vmap(lik.log_hessian_scalar))(y, t),
                [Y_st, T_f[..., 0], model.likelihood.likelihood_arr],
                [2, 2, 0],
                dim = len(model.likelihood.likelihood_arr),
                out_dim=1,
                batch_type = get_batch_type(model.likelihood.likelihood_arr)
            )
            hess = np.array(hess)
            hess = np.transpose(hess, [1, 2, 0]) # [Nt, Ns, P]
            neg_Lambda = hess[..., None] # [Nt, Ns, P, 1]
        else:
            # batch over Nt and Ns, only evaluate the conditional var on the indiviual P x 1 outputs
            Lambda = jax.vmap( jax.vmap( lambda f: model.likelihood.conditional_var(f) ))(T_f)

            # laplace approximation of the hessian
            neg_Lambda = -(1/Lambda)
            neg_Lambda = neg_Lambda[..., 0, 0] # will be [Nt x Ns x P x 1]

            # ensure rank 4
            if len(neg_Lambda.shape) == 3:
                # this is required due to an inconsistency in return dimensions
                #  when wrapping a likelihood in a ProductLikelihood
                neg_Lambda = neg_Lambda[..., None]

    else:
        q = model.approximate_posterior
        prior = model.prior
        Y = model.data.Y

        # N x P x 1
        chex.assert_rank(T_f, 3)

        # [N x P x B]
        if not laplace_log_lik:
            hess = batch_or_loop(
                lambda y, t, lik: jax.vmap(lik.log_hessian_scalar)(y, t),
                [Y.T, T_f[..., 0].T, model.likelihood.likelihood_arr],
                [0, 0, 0],
                dim = len(model.likelihood.likelihood_arr),
                out_dim=1,
                batch_type = get_batch_type(model.likelihood.likelihood_arr)
            )
            neg_Lambda = np.array(hess)
            neg_Lambda = (neg_Lambda.T)[..., None]
        else:
            # TODO: only works for exponential family likelihoods atm

            Lambda = jax.vmap(
                lambda f: model.likelihood.conditional_var(f[None, :]) # []
            )(
                T_f[..., 0] # [N x P]
            )

            Lambda = Lambda[:, :, 0, 0]
            # laplace approximation of the hessian
            neg_Lambda = -(1/Lambda)
            
        neg_Lambda = np.reshape(neg_Lambda, Y.shape)

        # Mask out entries corresponding to missing observations
        # These should just be ignored from the sums
        # N x P
        Y_mask = get_same_shape_mask(Y)
        chex.assert_equal(neg_Lambda.shape, Y_mask.shape)
        # N x P 
        neg_Lambda = neg_Lambda * Y_mask

        # N x P x 1
        neg_Lambda = neg_Lambda[..., None]

    return neg_Lambda

def compute_u_to_f(m, q_m, q_S, return_var_only = False, data = None):
    """
    Compute q(f) = E_p(f | u) [q(u | q_m, q_S)]

    Args:
        m: model
        q_m: M x Q x 1
        q_S: M x 1 x Q x Q
    """
    likelihood = m.likelihood
    prior = m.prior
    approximate_posterior = m.approximate_posterior
    inference = m.inference

    if data is None:
        data = m.data

    # compute the marginal q(f)
    q_f_mu, q_f_var = evoke('marginal', approximate_posterior, likelihood, prior, whiten=inference.whiten, debug=False)(
        data, q_m, q_S, approximate_posterior, likelihood, prior, inference.whiten
    )

    # If the model is Multioutput q_f_mu will be a list and each element of the list
    #   will have rank [3] and [4]

    if not(type(q_f_mu) is list):
        chex.assert_rank([q_f_mu, q_f_var], [3, 4])
        q_f_mu = [q_f_mu]
        q_f_var = [q_f_var]

    if return_var_only:
        return q_f_var

    return q_f_mu, q_f_var

def compute_f_to_tf(m, q_f_mu):
    """
    Computes an evaluation of T(F) 

    Args:
        q_f_mu: Shape N X Q x B or [(N X Q x B)]

    Output:
        q_f_mu: Shape  N x P x B
    """
    data = m.data

    # transform through non linear part
    if type(m.prior) == MultiOutput:
        q_f_res = []

        # transform each output separately
        # assumes that each output is a single output
        for i, p in enumerate(m.prior.parent):
            t_p = _process_samples(q_f_mu[i], lambda x:x, p)
            # TODO: only support single output transforms
            #q_f_res.append(np.squeeze(t_p))
            q_f_res.append(t_p[..., 0, 0])

        # fix shapes
        q_f_res = np.array(q_f_res).T
        chex.assert_rank(q_f_res, 2)

        q_f_res = q_f_res[..., None]
    else:
        q_f_res = _process_samples(q_f_mu[0], lambda x:x, m.prior)

    chex.assert_rank(q_f_res, 3)

    return q_f_res

def compute_u_to_tf(model, q_mu_z, q_var_z, data=None):
    """ Helper function to compute the transformations of u to T(F) """

    if model.approximate_posterior.meanfield_over_data:
        breakpoint()

    # compute u -> f
    q_f_mu, q_f_var = compute_u_to_f(model, q_mu_z, q_var_z, data=data)

    # compute f -> T(f)
    T_f = compute_f_to_tf(model, q_f_mu)
    chex.assert_rank(T_f, 3)

    return T_f

def _get_shape_conditional_f(model, m, S, data, X_st, Y_st) -> list:
    """
    For monte-carlo sampling we need the shapes of p(f|u) so we generate the white samples.
    We always return a list so we can support multiple outputs of different dimensions
    """

    # passing through S*0 makes means we only return the condition variance p(f|u) not the marginal q(f)
    # THIS IS JUST TO GET SHAPES -- VERY HACKY
    conditional_mean, conditional_var = compute_u_to_f(
        model, 
        m[0][None, ...], 
        S[0][None, ...]*0.0, 
        data=create_new_single_time_data_of_same_type(
            data, 
            X=X_st[0][None, ...], 
            Y=Y_st[0][None, ...]
        )
    )

    if type(conditional_mean) is not list:
        conditional_mean = [conditional_mean]
    else:
        if False:
            # the only time we will get a list is with MultiOutput, so wrap all other cases so they can be handled in the same way
            if type(model.prior) is not MultiOutput:
                conditional_mean = [conditional_mean]

    conditional_mean_shape = [
        a.shape for a in  conditional_mean
    ]

    return conditional_mean_shape

def _reparamaterise_f_to_tf(model, eps, pred_mu, pred_var):
    """
    Args:
        eps:
        pred_mu:
        pred_var:
    """
    P = len(eps)

    # should all be [Ns x Q x B]
    [chex.assert_equal_shape([pred_mu[p], eps[p]]) for p in range(P)]

    f_samples = []
    for p in range(P):
        pred_var_p = np.array(pred_var[p]) #Ns x B x Q x Q
        pred_var_p_chol = np.linalg.cholesky(pred_var_p) #Ns x Q x B x B

        eps_p = eps[p][..., None] #Ns x Q x B x 1
        eps_p = np.transpose(eps_p, [0, 2, 1, 3]) #Ns x B x Q x 1
        pred_mu_p = np.array(pred_mu[p])[..., None] #Ns x Q x B x 1
        pred_mu_p = np.transpose(pred_mu_p, [0, 2, 1, 3]) #Ns x B x Q x 1


        f_samples_p = pred_mu_p + np.nan_to_num(pred_var_p_chol * eps_p) # reparam trick -- Ns x B x Q x 1 
        f_samples_p = f_samples_p[..., 0] #Ns x B x Q 
        f_samples_p = np.transpose(f_samples_p, [0, 2, 1]) #Ns x Q x 1
        f_samples.append(f_samples_p)

    # should all be [Ns x Q x B]
    [chex.assert_equal_shape([pred_mu[p], f_samples[p]]) for p in range(P)]
    return compute_f_to_tf( model, f_samples )

def _reparamaterise_f_to_tf_across_time(model, eps, pred_mu, pred_var):
    P = len(eps)
    # vmap over time for arbtrary size lists of the same shape
    f_samples =  jax.vmap(
        lambda *x: _reparamaterise_f_to_tf(model, x[0:P], x[P:P*2], x[P*2:P*3]),
        [0] * P + [0] * P + [0] * P, 
    )(*eps, *pred_mu, *pred_var)

    return f_samples


def _f_conditional_samples(m, eps, model, S, X_st, Y_st, laplace_log_lik=False):
    """
    Args: 
        eps: sample of size [Nt x Ns x  Q x 1] 
        m: [Nt x (Ns x Q) x 1] 
        X_st: Nt x Ns x D
        Y_st: Nt x P x Ns if not TemporallyGrouped else Nt x  Ns x P
    """
    data = model.data

    def J_u_2_tf_wrapper(m_t, eps_t, S_t, x_t, y_t):
        P = len(eps_t)
        pred_mu_t, pred_var_t = compute_u_to_f(
            model, 
            m_t[None, ...], 
            S_t[None, ...]*0.0,  
            data=create_new_single_time_data_of_same_type(
                data, 
                X=x_t[None, ...], 
                Y=y_t[None, ...]
            )
        )
        return _reparamaterise_f_to_tf(model, eps_t, pred_mu_t, pred_var_t)

    # Nt x Ns x P x 1 x Ms x 1
    J_u_tf = jax.vmap(
        lambda m_t, S_t, x_t, y_t, *eps_t: jax.jacfwd(
            J_u_2_tf_wrapper, 
            argnums=[0]
        )(
            m_t, eps_t, S_t, x_t, y_t
        ),
        [0, 0, 0, 0] + [0]*len(eps) 
    )(
        m, S, X_st, Y_st, *eps
    )


    #u_to_f_mu is [Nt x Ns x Q x B]
    #u_to_f_var is [Nt x Ns x B x Q x Q]
    u_to_f_mu, u_to_f_var = jax.vmap(
        lambda m_t, S_t, x_t, y_t: compute_u_to_f(model, m_t[None, ...], S_t[None, ...], data=create_new_single_time_data_of_same_type(data, X=x_t[None, ...], Y=y_t[None, ...]))
    )(m, S, X_st, Y_st)


    Tf_sample = _reparamaterise_f_to_tf_across_time(model, eps, u_to_f_mu, u_to_f_var)

    neg_Lambda = get_likelihood_hessian(
        model, 
        Tf_sample, # Nt x Ns x P x 1
        laplace_log_lik=laplace_log_lik
    )

    # clean up shapes
    J_u_tf = J_u_tf[0]
    J_u_tf = J_u_tf[:, :, :, 0, :, :] # Nt x Ns x P x Ms x 1
    neg_Lambda = neg_Lambda[..., None] # Nt x Ns x P x 1 x 1

    if _ensure_str(data) == 'TemporallyGroupedData':
        Y_st_mask = get_same_shape_mask(Y_st) # Nt x Ns x P 
    else:
        Y_st_mask = get_same_shape_mask(Y_st) # Nt x P x Ns 
        Y_st_mask = np.transpose(Y_st_mask, [0, 2, 1]) # Nt x Ns x P 


    # create masks for missing lieklihoods
    neg_Lambda = neg_Lambda * Y_st_mask[..., None, None]

    # Gauss Newton approximation
    # TODO: should probably write as a jax.vjp
    # Nt x Ns x P x Ms x Ms
    G_vec = jax.vmap( # batch over time
        jax.vmap( #batch over space
            jax.vmap( #batch over outputs
                lambda a, b: a @ b @ a.T
            ) 
        )
    )(
        J_u_tf, neg_Lambda
    )

    # create masks for missing data
    Y_reshaped_for_G_mask = Y_st_mask[..., None, None] # Nt x Ns x P x 1 x1
    G_mask = np.tile( Y_reshaped_for_G_mask, [1, 1, 1, G_vec.shape[-2], G_vec.shape[-1]])
    chex.assert_equal(G_mask.shape, G_vec.shape)

    # remove missing data from the natural gradient sum
    G_vec_masked = G_mask * G_vec

    G = np.sum(G_vec_masked, [1, 2]) # Nt x Ms x Ms
    G = G[:, None, ...] # Nt x 1 x Ms x Ms

    if settings.verbose:
        print('ST GAUSS NEWTON')

    if data.minibatch:
        G = G*data.minibatch_scaling

    return G

def gauss_newton_jacobian_approximation_across_time(u, S, model,  laplace_log_lik=False, prediction_samples=None, delta_f=True):
    """
    Gaussian newton approximation whilst exploiting spatio-temporal structure in the data
    """
    m = u

    q = model.approximate_posterior
    prior = model.prior
    Y = model.data.Y

    data = model.data

    # Nt x P x Ns format
    X_st, Y_st = data.X_st, data.Y_st

    conditional_f_shape: list = _get_shape_conditional_f(model, m, S, data, X_st, Y_st)

    if _ensure_str(data) == 'TemporallyGroupedData':
        sample_shape = [
            [Y_st.shape[0], Y_st.shape[1], s[1], s[2]] # Nt x Ns x Q x B
            for s in conditional_f_shape
        ]
    else:
        sample_shape = [
            [Y_st.shape[0], Y_st.shape[2], s[1], s[2]] # Nt x Ns x Q x B
            for s in conditional_f_shape
        ]

    if not delta_f:
        if settings.verbose:
            print('NG: monte carlo approx for expectation over p(f|u)')

        white_samples = [
            objax.random.normal(
                [settings.ng_f_samples] + s, 
                mean=0.0, 
                stddev=1.0, 
                generator= model.inference.generator
            )
            for s in sample_shape
        ]

        G_over_samples = jax.vmap(
            lambda *sample: _f_conditional_samples(m, sample, model, S, X_st, Y_st, laplace_log_lik=laplace_log_lik)
        )(*white_samples)

        G = np.mean(G_over_samples, axis=0)
    else:
        if settings.verbose:
            print('NG: delta method for expectation over p(f|u)')
        # passing through a sample of zero means that all covariance terms will cancel and we will just have the mean, hence the delta method
        G = _f_conditional_samples(m, [np.zeros(sample_shape[p]) for p in range(len(sample_shape))],  model, S, X_st, Y_st, laplace_log_lik=laplace_log_lik)

    return G


def gauss_newton_jacobian_approximation(u, S, model,  laplace_log_lik=False, prediction_samples=None, delta_f=True):
    """
    Gaussian newton approximation without exploiting spatio-temporal structure in the data. This is inefficient but simple to implement
        so useful for testing and debugging purposes.
    """
    if delta_f is False:
        raise NotImplementedError('delta_f = False is only implemented with data that decomposes across time!')
    m = u
    q = model.approximate_posterior
    prior = model.prior
    Y = model.data.Y

    u_tf_fn = lambda m: compute_u_to_tf(model, m, S)
    J_u_tf = jax.jacfwd(u_tf_fn)(m)

    neg_Lambda = get_likelihood_hessian(model, u_tf_fn(m), laplace_log_lik=laplace_log_lik)

    # sum over B?
    G_vec = jax.vmap(
        # sum over N
        lambda Ju: jax.vmap(
                # sum over P/Q
                jax.vmap(
                    lambda a, b: a @ b @ a.T
                )
            )(
                Ju[:, :, None, ...], 
                neg_Lambda[..., None]
            ),
        3
    )(J_u_tf)

    Y_mask = get_same_shape_mask(Y)
    Y_reshaped_for_G_mask = Y_mask[None, ...][..., None, None, None, None] 

    G_mask = np.tile( Y_reshaped_for_G_mask, [G_vec.shape[0], 1, 1, 1, G_vec.shape[4], G_vec.shape[5], G_vec.shape[6]])
    chex.assert_equal(G_mask.shape, G_vec.shape)

    # remove missing data from the natural gradient sum
    G_vec_masked = G_mask * G_vec

    G = np.sum(G_vec_masked, [1, 2, 3, 6])[:, None, ...]

    if model.data.minibatch:
        G = G*model.data.minibatch_scaling

    if settings.verbose:
        print('D GAUSS NEWTON')

    return G



def gauss_newton(u, S, model,  laplace_log_lik=False, prediction_samples=None, delta_f=True):
    """
    Args:
        u: Nt x Ms x D
    """
    chex.assert_rank([u, S], [3, 4])

    q = model.approximate_posterior
    prior = model.prior
    Y = model.data.Y

    if settings.cvi_ng_batch:
        # only minibatch once so everything is computed with the same batch
        if model.data.minibatch:
            # TODO: minibatching only works when sparsity is used. Assert this.
            model.data.batch()
    else:
        #do not batch so we use the same batch as used in teh ELBO computation
        pass

    if data_decomposes_across_time(model.data):
        G = gauss_newton_jacobian_approximation_across_time(u, S, model,  laplace_log_lik=laplace_log_lik, prediction_samples=prediction_samples, delta_f=delta_f)

        if False:
            G_d = gauss_newton_jacobian_approximation(u, S, model,  laplace_log_lik=laplace_log_lik, prediction_samples=prediction_samples, delta_f=delta_f)
            print(G-G_d)
            print(np.sum(G-G_d))
            print(np.sum(np.abs(G-G_d)))
            breakpoint()
    else:
        G = gauss_newton_jacobian_approximation(u, S, model,  laplace_log_lik=laplace_log_lik, prediction_samples=prediction_samples, delta_f=delta_f)

    approx_hessian = 0.5 *  G
    chex.assert_rank(approx_hessian, 4)

    return approx_hessian

def laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=True, delta_u = True, delta_f = True):
    q = model.approximate_posterior


    if _ensure_str(q) == 'MeanFieldConjugateGaussian':
        # block diagonal

        approx_posteriors = q.approx_posteriors
        q_mu_z, q_var_z = batch_or_loop(
            lambda q: q.surrogate.posterior_blocks(),
            [approx_posteriors],
            [0],
            dim = len(approx_posteriors),
            out_dim=2,
            batch_type = get_batch_type(approx_posteriors)
        )

        # fix shapes
        Q, N, L, B = q_mu_z.shape
        q_mu_z = np.transpose(q_mu_z, [1, 0, 2, 3])
        q_mu_z = np.reshape(q_mu_z, [N, Q*L, B])

        q_var_z = np.transpose(q_var_z[:, :, 0, ...], [1, 0, 2, 3])
        #q_var_z = jax.vmap(to_block_diag)(q_var_z)
        #q_var_z = q_var_z[:, None, ...]
        
    else:
        # get parameters of q(u) in time-latent-space order
        q_mu_z, q_var_z = q.surrogate.posterior_blocks()
        chex.assert_rank([q_mu_z, q_var_z], [3, 4])

    # delta u
    if delta_u:
        approx_hessian = gauss_newton(q_mu_z, q_var_z , model, laplace_log_lik=laplace_log_lik, prediction_samples=prediction_samples, delta_f=delta_f)
    else:
        def wrapped_fn(s):
            return  gauss_newton(s, q_var_z , model, laplace_log_lik=laplace_log_lik, prediction_samples=prediction_samples, delta_f=delta_f)

        # sample u here
        approx_hessian = mv_block_monte_carlo(
            wrapped_fn, 
            q_mu_z, 
            q_var_z, 
            generator = model.inference.generator, 
            num_samples = prediction_samples
        )

    if _ensure_str(q) == 'MeanFieldConjugateGaussian':
        # fix shapes
        approx_hessian = jax.vmap(to_block_diag)(approx_hessian)
        approx_hessian = approx_hessian[:, None, ...]
    chex.assert_rank(approx_hessian, 4)
    return approx_hessian


def get_full_gaussian_hessian_approximation(model, beta, prediction_samples, enforce_psd_type):
    if enforce_psd_type == 'gauss_newton':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=False, delta_u = False, delta_f = True)
    elif enforce_psd_type == 'gauss_newton_delta_u':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=False, delta_u = True, delta_f = True)
    elif enforce_psd_type == 'laplace_gauss_newton':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=True, delta_u = False, delta_f = True)
    elif enforce_psd_type == 'laplace_gauss_newton_delta_u':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=True, delta_u = True, delta_f = True)
    elif enforce_psd_type == 'gauss_newton_mc_f':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=False, delta_u = False, delta_f = False)
    elif enforce_psd_type == 'gauss_newton_delta_u_mc_f':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=False, delta_u = True, delta_f = False)
    elif enforce_psd_type == 'laplace_gauss_newton_mc_f':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=True, delta_u = False, delta_f = False)
    elif enforce_psd_type == 'laplace_gauss_newton_delta_u_mc_f':
        approx_hessian =  laplace_gauss_newton_natural_gradient_for_full_gaussian_approx_posterior(model, beta, prediction_samples, laplace_log_lik=True, delta_u = True, delta_f = False)
    else:
        raise RuntimeError()

    chex.assert_rank(approx_hessian, 4)
    return approx_hessian








