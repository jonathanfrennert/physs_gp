import chex
import jax
import jax.numpy as np
import objax

from ...dispatch import dispatch, evoke, DispatchNotFound
from ..matrix_ops import block_from_vec, block_from_mat, stack_rows, shape_rank

from ...data import Data, TransformedData
from ...transforms import LinearTransform, Independent, NonLinearTransform, Transform, DataLatentPermutation, MultiOutput
from ...utils.batch_utils import batch_over_module_types
from ...utils.nan_utils import get_mask, mask_vector, mask_matrix, get_same_shape_mask
from ...utils.utils import get_batch_type
from ...likelihood import ProductLikelihood, DiagonalLikelihood, Likelihood, DiagonalGaussian, Gaussian, BlockDiagonalGaussian, GaussianProductLikelihood, PowerLikelihood
from .expected_log_likelihoods import scalar_gaussian_expected_log_likelihood, gaussian_expected_log_likelihood, full_gaussian_expected_log_likelihood, full_gaussian_expected_log_precision_likelihood
from ..integrals.approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ..integrals.samples import approximate_expectation
from ...approximate_posteriors import MeanFieldApproximatePosterior, GaussianApproximatePosterior, FullGaussianApproximatePosterior, ApproximatePosterior
from ...core.model_types import get_model_type, LinearModel, NonLinearModel, get_non_linear_model_part
from ...core.block_types import get_block_type, compare_block_types, Block
from ...core.gp_prior import GPPrior
from ..permutations import  permute_mat
from ... import settings
from ..parameter_transforms import softplus

from batchjax import batch_or_loop, BatchType
from numpy.polynomial.hermite import hermgauss
from jax import lax



# ====================== SCALAR ELL COMPONENTS ===================
@dispatch(Gaussian)
def scalar_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    """ Gaussian expected log likelihood component. """
    return scalar_gaussian_expected_log_likelihood(X, Y, likelihood.variance, q_f_mu, q_f_var)

# TODO: rename this
@dispatch(GaussianProductLikelihood, Block.BLOCK)
def element_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood):
    lik_var = np.diag(likelihood.variance)
    return full_gaussian_expected_log_likelihood(X, Y, lik_var, q_f_mu, q_f_var)


# ====================== GAUSSIAN ELLs ===================

@dispatch(BlockDiagonalGaussian, Block.BLOCK)
@dispatch(BlockDiagonalGaussian, Block.DIAGONAL)
def single_output_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, block_type):
    """
    Blocked Gaussian Likelihood

    This can occur in CVI when using:
        multiple outputs
        spatial-temporal data
        aggregated data
    """
    # TODO: need to collapse/join the P and B axis

    # TODO: adding missing data masking
    block_size = likelihood.block_size
    #num_blocks = q_f_mu.shape[0]
    num_blocks = likelihood.num_blocks

    # ensure Y is of the correct shape
    X_blocks = np.reshape(X, [num_blocks, -1, X.shape[-1]])

    # Y is in (time-space)-latent format
    # reshape to time - (space-latent) format
    Y = np.reshape(Y, [num_blocks, block_size, 1])

    # already in time - (space-latent) format 
    q_f_mu = np.reshape(q_f_mu, [num_blocks, block_size, 1])

    N = Y.shape[0]

    # ensure Y is 3d
    if len(Y.shape) == 2:
        Y = Y[..., None]

    if len(q_f_mu.shape) == 2:
        # q_f_mu across all ouputs in N x P x B, but we have batched over P so now it is rank 2
        # add back the missing axis so that is matched Y
        q_f_mu = q_f_mu[:, None, :]

    # ensure q_f_var is 3d
    if len(q_f_var.shape) == 4:
        q_f_var = q_f_var[:, 0, : ,:]


    # ensure correct shapes
    chex.assert_rank([Y, q_f_mu, q_f_var], [3, 3, 3])
    chex.assert_equal(Y.shape, q_f_mu.shape)
    chex.assert_shape(q_f_var, [N, block_size, block_size])

    # block X
    if False:
        X_blocks = np.tile(X[None, ...], [block_size, 1, 1])
        X_blocks = np.transpose(X_blocks, [1, 0, 2])

    fn = full_gaussian_expected_log_likelihood

    if str(type(likelihood).__name__) == 'PrecisionBlockDiagonalGaussian':
        fn = full_gaussian_expected_log_precision_likelihood

        lik_inv = likelihood.precision
        chex.assert_shape(lik_inv, q_f_var.shape)


        # convert to time - space - latent format
        lik_inv = jax.vmap(lambda A: permute_mat(A, likelihood.num_latents))(lik_inv)
        lik_mat = lik_inv
    else:
        # likelihood is time - latent - space format
        lik_var = likelihood.variance
        chex.assert_shape(lik_var, q_f_var.shape)

        # convert to time - space - latent format
        lik_var = jax.vmap(lambda A: permute_mat(A, likelihood.num_latents))(lik_var)
        lik_mat = lik_var

    ell_arr = jax.vmap(
        fn,
        [0, 0, 0, 0, 0],
        0
    )(X_blocks, Y, lik_mat, q_f_mu, q_f_var)
    #breakpoint()


    ell = np.sum(ell_arr)

    return ell



@dispatch("DiagonalGaussian", Block.DIAGONAL)
def single_output_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, block_type):
    lik_var = likelihood.variance_param.value

    chex.assert_rank([lik_var, Y, q_f_mu, q_f_var], [1, 2, 2, 3])

    # q_f_var is diagional so we fix the shapes so all shapes batch
    q_f_var = q_f_var[..., 0]
    chex.assert_equal_shape([Y, q_f_mu, q_f_var])

    # Get nan mask for output
    mask = get_mask(Y)

    # Convert nans to zeros
    Y = mask_vector(Y, mask)

    # Ensure rank 2 after batching
    X = X[:, None, ...]
    Y = Y[..., None]
    q_f_mu = q_f_mu[..., None]
    q_f_var = q_f_var[..., None]

    # Compute ELL for each datapoint
    ell_arr = jax.vmap(
        scalar_gaussian_expected_log_likelihood,
        [0, 0, 0, 0, 0],
        0
    )(X, Y, lik_var, q_f_mu, q_f_var)

    # Set elements that correposnd to missing data to zero
    ell_arr = mask_vector(ell_arr[:, None], mask)

    # Only sums the ELL terms without missing data
    ell = np.sum(ell_arr)

    return ell

@dispatch(DiagonalLikelihood, Block.DIAGONAL)
def single_output_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, block_type):
    """ For diagonal likelihoods adds support for missing data. """ 
    chex.assert_rank([Y, q_f_mu, q_f_var], [2, 2, 3])

    # q_f_var is diagional so we fix the shapes so all shapes batch
    q_f_var = q_f_var[..., 0]
    chex.assert_equal_shape([Y, q_f_mu, q_f_var])

    # Get nan mask for output
    mask = get_mask(Y)

    # Convert nans to zeros
    Y = mask_vector(Y, mask)

    # Ensure rank 2 after batching
    X = X[:, None, ...]
    Y = Y[..., None]
    q_f_mu = q_f_mu[..., None]
    q_f_var = q_f_var[..., None]


    fn = evoke('scalar_expected_log_likelihood', likelihood)

    # Compute ELL for each datapoint
    ell_arr = jax.vmap(
        fn,
        [0, 0, 0, 0, None],
        0
    )(X, Y, q_f_mu, q_f_var, likelihood)

    # Set elements that correposnd to missing data to zero
    ell_arr = mask_vector(ell_arr[:, None], mask)

    # Only sums the ELL terms without missing data
    ell = np.sum(ell_arr)

    return ell

@dispatch(PowerLikelihood, Block.DIAGONAL)
@dispatch(PowerLikelihood, Block.BLOCK)
def single_output_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, block_type):
    scale_a = likelihood.a

    parent_lik = likelihood.parent

    parent_ell =  evoke('expected_log_likelihood', parent_lik, block_type)(
        X, Y, q_f_mu, q_f_var, parent_lik, block_type
    )

    return parent_ell * scale_a


@dispatch(ProductLikelihood, Block.BLOCK)
def single_output_expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, block_type):
    chex.assert_rank([Y, q_f_mu, q_f_var], [2, 3, 4])

    # As this ELL does not decompose across latents and datapoints
    #   the nan-handling must be handled lower down

    # Ensure rank 2 after batching
    X = X[:, None, ...]
    Y = Y[..., None]
    q_f_var = q_f_var[:, 0, ...]

    fn = evoke('element_expected_log_likelihood', likelihood, block_type)

    # Compute ELL for each datapoint
    ell_arr = jax.vmap(
        fn,
        [0, 0, 0, 0, None],
        0
    )(X, Y, q_f_mu, q_f_var, likelihood)


    # Only sums the ELL terms without missing data
    ell = np.sum(ell_arr)

    return ell

# ====================== ELL FOR DIFFERENT APPROXIMATE POSTERIORS ===================

def compute_ell_for_sample(transformed_f, X, Y, prior, likelihood, approximate_posterior):
    """
    Args:
        transformed_f: (N x P x B) or (N x P x B x B) - sampled  and transformed f
        Y: N x P or N x B x P


    When Y is of rank 2 it is assumed that there is one likelihood per output (as defined by the second axis) 
       
    When Y is of rank 3 (or greater) it is assumed that there is one likelihood per grouping, as defined by the first axis

    """

    # assert correct shapes
    # transformed_f should have at least rank 3 (N x P x B)
    if False:
        if shape_rank(transformed_f) < 3:
            raise RuntimeError('transformed_f shape wrong')

        if shape_rank(Y) <= 2:
            raise RuntimeError('Shape of Y is wrong')

    multivariate_flag = False
    # fix shapes of Y to cover regression, multi output, covariance regression etc
    if type(Y) is not list:
    #if shape_rank(Y) == 2:
        # Y and F must be rank 2 when they are passed to log_likelihood
        # When vmapping one dimension is lost so extent here
        Y = (Y.T)[..., None]
    #elif shape_rank(Y) >= 2:
    else:
        multivariate_flag = True

    N, Q, B = transformed_f.shape[0], transformed_f.shape[1], transformed_f.shape[2]
    
    # P is the number of likelihoods
    #P = Y.shape[0]

    likelihood_arr = likelihood.likelihood_arr
    num_likelihoods = len(likelihood_arr)

    #chex.assert_equal(P, num_likelihoods)

    # Y and F must be rank 2 when they are passed to log_likelihood
    # When vmapping one dimension is lost so extent here
    #Y = Y[..., None]
    #chex.assert_shape(transformed_f, Y.shape)

    # TODO: fix this -- we are currently assuming that likelihood is a product likelihood 
    # when Y is a single output we can simply mask by ignoring the corresponding ELL for each datapoint
    # otherwise we have to let the likelihood handle it

    # Get nan mask for output
    raw_Y = Y
    mask = get_same_shape_mask(Y)

    # Convert nans to zeros
    Y = mask_matrix(Y, mask)

    # batch over outputs
    # log likelihood for each outout
    if settings.experimental_allow_f_multi_dim_per_output:
        # TODO: dimensions might go funny here...
        ll_arr = [ np.squeeze(likelihood_arr[0].log_likelihood(Y[0], transformed_f)) ]
    else:
        ll_arr = batch_or_loop(
            lambda y, f, lik: np.squeeze(lik.log_likelihood(y, f)),
            [Y, transformed_f, likelihood_arr],
            [0, 1, 0],
            dim = num_likelihoods,
            out_dim=1,
            batch_type = get_batch_type(likelihood_arr)
        )
    # P x N x B
    ll_arr = np.array(ll_arr)
    # Fix shapes so that ll_arr matches Y
    ll_arr = ll_arr[..., None]
    chex.assert_equal(ll_arr.shape, Y.shape)
    # Mask out log-liklihoods that correspond to missing data
    ll_arr = mask_matrix(ll_arr, mask)
    ll_arr = np.transpose(ll_arr, [1, 0, 2])

    return ll_arr

@dispatch(Data, Likelihood, 'GPPrior', ApproximatePosterior, Block.DIAGONAL)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, prior, approximate_posterior, inference, block_type):
    chex.assert_rank([q_f_mu, q_f_var], [3, 4])
    chex.assert_equal([q_f_mu.shape[1], q_f_var.shape[1]], [1, 1])

    if False: 
        # use monte carlo / quadrature
        pass
    else:
        # single output already 
        ell = evoke('single_output_expected_log_likelihood', likelihood, block_type)(
            X, Y, q_f_mu[:, 0, ...], q_f_var[:, 0, ...], likelihood, block_type
        )

        return ell


@dispatch(Data, ProductLikelihood, Transform, ApproximatePosterior, Block.DIAGONAL)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, prior, approximate_posterior, inference, block_type):
    chex.assert_rank([q_f_mu, q_f_var], [3, 4])

    N, P, B = q_f_mu.shape

    chex.assert_equal([q_f_mu.shape[0], q_f_mu.shape[1]] , [N, P])
    chex.assert_equal([q_f_var.shape[0], q_f_var.shape[1]] , [N, P])


    model_type = get_model_type(prior)

    # TODO: this is not very general, fix this at some point
    gauss_lik_flag = False
    if isinstance(likelihood, ProductLikelihood):
        # check if all likelihood_arr are Gaussian
        gauss_lik_flag = all([isinstance(lik, Gaussian) or isinstance(lik, DiagonalGaussian)  for lik in likelihood.likelihood_arr])
    elif isinstance(likelihood, BlockDiagonalGaussian):
        gauss_lik_flag = True


    # TODO: there is a choice here between quadrature and monte-carlo estimation
    # TODO: need to check if a likelihood has a closed form ELL
    if isinstance(model_type, LinearModel) and gauss_lik_flag:
        # check if closed form expression exists
        # batch over each output
        likelihood_arr = likelihood.likelihood_arr

        block_arr = [block_type for l in likelihood_arr]

        # Ensure rank 2 after batching
        Y = Y[..., None]
        ell_arr = batch_over_module_types(
            evoke_name = 'single_output_expected_log_likelihood',
            evoke_params = [],
            module_arr = [likelihood_arr, block_arr],
            fn_params = [X, Y, q_f_mu, q_f_var, likelihood_arr, block_arr],
            fn_axes = [None, 1, 1, 1, 0, 0],
            dim = P,
            out_dim  = 1 
        )
        #ensure array
        ell_arr = np.array(ell_arr)

        chex.assert_shape(ell_arr, [P])

        ell = np.sum(ell_arr)

        return ell

    else:
        likelihood_arr = likelihood.likelihood_arr

        # q_f_mu is of shape:
        #   N x P x B
        # q_v_var is of shape:
        #   N x P x B x B or N x 1 x PB x PB

        num_likelihoods = len(likelihood_arr)
        N, P = Y.shape
        Q = prior.base_prior.output_dim

        ell = approximate_expectation(
            compute_ell_for_sample, 
            q_f_mu, 
            q_f_var, 
            prior = prior,
            fn_args = [X, Y, prior, likelihood, approximate_posterior],
            generator = inference.generator, 
            num_samples = inference.ell_samples,
            block_type = block_type,
            average = True
        )

        chex.assert_shape(ell, [N, P, 1])

        ell = np.sum(ell)

        return ell

    raise RuntimeError()

@dispatch(Data, Likelihood, 'GPPrior', ApproximatePosterior, Block.BLOCK)
@dispatch(Data, Likelihood, Transform, ApproximatePosterior, Block.BLOCK)
def expected_log_likelihood(X, Y, q_f_mu, q_f_var, likelihood, prior, approximate_posterior, inference, block_type):
    chex.assert_rank([q_f_mu, q_f_var], [3, 4])
    chex.assert_equal([q_f_var.shape[1]], [1])

    N, Q, B = q_f_mu.shape
    #P = Y.shape[1]

    model_type = get_model_type(prior)

    try:
        # if model is non linear then there will be no closed form expressions
        if isinstance(model_type, NonLinearModel):
            raise DispatchNotFound('Nonlinear Model Required ELL approximation')

        # see if there is a closed form expression
        ell = evoke('single_output_expected_log_likelihood', likelihood, block_type)(
           X, Y, q_f_mu, q_f_var, likelihood, block_type
        )

        ell = np.sum(ell)

        return ell
    except DispatchNotFound as e:
        # approximate expected log likelihood
        ell = approximate_expectation(
            compute_ell_for_sample, 
            q_f_mu, 
            q_f_var, 
            prior = prior,
            fn_args = [X, Y, prior, likelihood, approximate_posterior],
            generator = inference.generator, 
            num_samples = inference.ell_samples,
            block_type = block_type,
            average = True
        )

        #chex.assert_shape(ell, [N, P, 1])

        if settings.experimental_simple_time_weight:
            if settings.verbose:
                print('using experimental_simple_time_weight')
            alpha = 1.0
            time_weight = alpha * ((np.max(X[:, 0])-X[:, 0])+1)
            ell = time_weight[:, None, None]*np.array(ell)

        if settings.experimental_cumsum_time_weight:
            if settings.verbose:
                print('using experimental_cumsum_time_weight')

            # TODO: assuming ordered by time
            # group by time
            # use a precomputed segment_sum to make things jittable
            t_int = settings.experimental_precomputed_segements
            #_, t_int = np.unique(np.squeeze(t), return_inverse =True)
            ell_sum_by_time = jax.ops.segment_sum(
                np.squeeze(ell),
                t_int, 
                num_segments = settings.experimental_precomputed_num_segements
            )
            ell_cumsum = np.cumsum(ell_sum_by_time)[:-1]
            ell_cumsum = np.hstack([np.array([0]), ell_cumsum])
            ell_weights = softplus(settings.experimental_precomputed_cumsum_eps*np.clip(ell_cumsum*-1, a_max=0))
            ell =  ell_weights * ell_sum_by_time

        ell = np.sum(np.array(ell))

        return ell

    raise RuntimeError()

# ===============================================================================
# ================================= Specifics =================================
# ===============================================================================

@dispatch(Data, 'HetGaussian', Transform, MeanFieldApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """
    TODO: this is a hack for now 
    """
    def ell_scalar(y, f_mu, f_var):
        y = np.squeeze(y)
        f_mu = np.squeeze(f_mu)
        f_var = np.squeeze(f_var)

        m_f = f_mu[0]
        m_g = f_mu[1]
        k_f = f_var[0]
        k_g = f_var[1]

        return -0.5 * (
            np.log(2 * np.pi) +  m_g + ((y - m_f) ** 2 + k_f) * np.exp(0.5 * k_g -  m_g)
        )

    ell = jax.vmap(ell_scalar)(data.Y, q_f_mu_arr, q_f_var_arr)

    return np.sum(ell)

    X, Y = data.X, data.Y
    scalar_fn = lambda y, f: likelihood.log_likelihood_scalar(np.squeeze(y), np.squeeze(f))


    ell = approximate_expectation(
        lambda transformed_f, X, Y, prior, likelihood, approximate_posterior: jax.vmap(
           scalar_fn 
        )(Y, transformed_f), 
        q_f_mu_arr, 
        q_f_var_arr, 
        prior = prior,
        fn_args = [X, Y, prior, likelihood, approximate_posterior],
        generator = inference.generator, 
        num_samples = inference.ell_samples,
        block_type = Block.DIAGONAL,
        average = True
    )

    return ell

# ===============================================================================
# ================================= Entry Point =================================
# ===============================================================================

@dispatch(Data, Likelihood, 'GPPrior', ApproximatePosterior)
@dispatch(Data, Likelihood, Transform, ApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """
    General ELL entry point
    """
    chex.assert_rank([q_f_mu_arr, q_f_var_arr], [3, 4])
    base_prior = prior.base_prior

    # get correct ELL corresponding to the blocks
    q_block_size = q_f_var_arr.shape[-1]
    lik_block_type = likelihood.block_type


    block_type_p: Block = compare_block_types(
        lik_block_type, 
        get_block_type(q_block_size)
    )

    X = data.X
    Y = data.Y
    
    if False:
        try:
            ell_true = jax.vmap( full_gaussian_expected_log_likelihood, [None, 0, None, 0, 0])(X, Y[:, [0]][..., None], np.reshape(likelihood.likelihood_arr[0].variance, [1, 1]), q_f_mu_arr[:, 0, ...][..., None], q_f_var_arr[:, :, 0, 0][..., None])

            return np.sum(ell_true)
        except Exception as e:
            pass

    ell =  evoke('expected_log_likelihood', data, likelihood, prior, approximate_posterior, block_type_p)(
        X, Y, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference, block_type_p
    )

    #breakpoint()

    return ell

@dispatch(Data, ProductLikelihood, MultiOutput, MeanFieldApproximatePosterior)
@dispatch(Data, ProductLikelihood, MultiOutput, FullGaussianApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """
    Multioutput assumes that... 
    """
    ell_arr = []

    # TODO: what about data?
    num_likelihoods = len(likelihood.likelihood_arr)
    for p in range(num_likelihoods):
        prior_p = prior.parent[p]
        likelihood_p = likelihood.likelihood_arr[p]
        q_f_mu_p = q_f_mu_arr[p]
        q_f_var_p = q_f_var_arr[p]


        chex.assert_rank([q_f_mu_p, q_f_var_p], [3, 4])

        q_block_size = q_f_var_p.shape[-1]
        lik_block_type = likelihood.block_type

        X_p = data.X

        # This is a bit hacky, it is just a way to define which output of Y correspond to the outputs of q(f)
        # ideally data would be a data list and then it is specified through the data object, not the transform
        if prior_p.data_y_index is not None:
            Y_p = data.Y[:, prior_p.data_y_index]
        else:
            Y_p = data.Y[:, p][:, None]

        block_type_p: Block = compare_block_types(
            lik_block_type, 
            get_block_type(q_block_size)
        )

        #idx = np.squeeze(~np.isnan(Y_p))

        #return gaussian_expected_log_likelihood(X_p[idx, :], Y_p[idx, :], likelihood_p.likelihood_arr[0].variance, q_f_mu_p[..., 0][idx, :], q_f_var_p[:, :, 0, 0][idx, :])

        ell_p =  evoke('expected_log_likelihood', data, likelihood_p, prior_p, approximate_posterior, block_type_p)(
            X_p, Y_p, q_f_mu_p, q_f_var_p, likelihood_p, prior_p, approximate_posterior, inference, block_type_p
        )
        ell_arr.append(ell_p)

    if settings.verbose:
        print('ell_arr: ', ell_arr)

    return np.sum(np.array(ell_arr))

@dispatch(TransformedData, Likelihood, Transform, ApproximatePosterior)
def expected_log_likelihood(data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference):
    """ Transformed Y ELL. """

    base_data = data.base_data

    base_ell =  evoke('expected_log_likelihood', base_data, likelihood, prior, approximate_posterior)(
        data, q_f_mu_arr, q_f_var_arr, likelihood, prior, approximate_posterior, inference
    )

    # convert nans to zero
    # these will have a jacobian of zero and so will not contribute to the sum
    log_jac = data.log_jacobian(np.nan_to_num(data.Y_base, nan=1.0))

    # ensure it does not constribute
    nan_mask = get_same_shape_mask(data.Y_base)

    #ignore nans
    log_jac = nan_mask * log_jac

    log_jac = np.sum(log_jac)

    return base_ell + log_jac
