import jax
import jax.numpy as np
import objax
import chex
from ...import settings

from .approximators import mv_indepentdent_monte_carlo, mv_block_monte_carlo
from ...core.model_types import get_model_type, LinearModel, NonLinearModel, get_linear_model_part, get_non_linear_model_part

from ...core.block_types import get_block_type, compare_block_types, Block


def _process_samples(f, fn, prior, *args):
    """
    Args:
        f: N x P x B - sampled f
    """
    chex.assert_rank(f, 3)
    non_linear_prior_part = get_non_linear_model_part(prior)

    if non_linear_prior_part is not None:
        transformed_f = f
        for p in non_linear_prior_part:
            # Reparameterise
            # Transform through prior for each datapoint
            transformed_f = jax.vmap(
                p.forward,
                [0],
                0
            )(transformed_f)
    else:
        transformed_f = f

    return fn(transformed_f, *args)


def approximate_diagonal_expectation(fn, mu, var, prior, fn_args, num_samples, block_type: Block, generator, average):
    chex.assert_rank([mu, var], [3, 4])
    if var.shape[-1] == 1:
        # var is N x Q x 1 x 1
        # convert to N x Q x 1
        var = var[..., 0]
    else:
        # var is N x 1 x Q x Q
        # convert to N x Q x 1
        var = np.diagonal(var, axis1=2, axis2=3)
        var = np.transpose(var, [0, 2, 1])

    wrapped_fn = lambda f, *f_args: _process_samples(f, fn, prior, *f_args)

    if not settings.use_quadrature:
        if False:
            if num_samples == 1:
                return wrapped_fn(mu,  *fn_args)[None, ...]

        ell =  mv_indepentdent_monte_carlo(
            wrapped_fn,
            mu, 
            var, 
            fn_args = fn_args,
            generator = generator, 
            num_samples = num_samples,
            average = average
        )

        return ell
    else:
        num_quad_points = num_samples

        x, w = onp.hermgauss(num_quad_points)
        const = np.pi**-0.5

        q_f_var = np.squeeze(q_f_var)
        q_f_mu = np.squeeze(q_f_mu)
        Y = np.squeeze(Y)

        # change of variable
        f = 2.0**0.5*np.sqrt(q_f_var)*x + q_f_mu  

        chex.assert_shape(f, [num_quad_points])

        res = jax.vmap(
            likelihood.log_likelihood_scalar, 
            [None, 0], 
            0
        )(Y, f)

        chex.assert_shape(res, [num_quad_points])
        
        return np.sum(w * const*res)

def approximate_blocked_expectation(fn, mu, var, prior, fn_args, num_samples, block_type: Block, generator, average):

    wrapped_fn = lambda f, *f_args: _process_samples(f, fn, prior, *f_args)

    if not settings.use_quadrature:
        if False:
            if num_samples == 1:
                return wrapped_fn(mu,  *fn_args)[None, ...]

        samples = mv_block_monte_carlo(
            wrapped_fn,
            mu, 
            var, 
            fn_args = fn_args,
            generator = generator, 
            num_samples = num_samples,
            average = average
        )
    else:
        raise NotImplementedError()


    return samples

def approximate_expectation(fn, mu, var, prior, fn_args, num_samples = None, block_type: Block = None, generator = None, average=True):
    if block_type is None: raise RuntimeError('Block type must be passed')
    if num_samples is None: raise RuntimeError('Number of samples must be passed')

    if block_type == Block.DIAGONAL:
        return approximate_diagonal_expectation(fn, mu, var, prior, fn_args, num_samples, block_type, generator, average)

    return approximate_blocked_expectation(fn, mu, var, prior, fn_args, num_samples, block_type, generator, average)

