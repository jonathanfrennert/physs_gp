import jax
import jax.numpy as np
import objax
import chex
from ...import settings

def gauss_quad():
    pass

def mv_gauss_quad():
    pass

def monte_carlo():
    pass

def mv_indepentdent_monte_carlo(fn, mu_arr, var_arr, fn_args =[], generator=None, num_samples=100, average=True):
    """
    multi-variate monte-carlo 

    Args:
        fn: Callable - 
        mu_arr: N x P x B
        var_arr: N x P x B
    """

    if generator == None: raise RuntimeError()

    chex.assert_equal(mu_arr.shape, var_arr.shape)
    chex.assert_rank(mu_arr, 3)

    N, Q, B = mu_arr.shape

    white_samples = objax.random.normal([num_samples]+list(mu_arr.shape), mean=0.0, stddev=1.0, generator=generator)

    chex.assert_shape(white_samples, [num_samples, N, Q, B])

    # Reparameterise
    def reparameterise(fn, samples, mu_arr, var_arr, *args):
        chex.assert_equal(samples.shape, mu_arr.shape)
        chex.assert_equal(samples.shape, var_arr.shape)

        s = mu_arr + samples * np.sqrt(var_arr)
        return fn(s, *args)

    num_args = len(fn_args)

    # batch over samples
    fn_samples = jax.vmap(
        reparameterise,
        [None, 0, None, None] + [None]*num_args,
        0
    )(fn, white_samples, mu_arr, var_arr, *fn_args)

    if average:
        # Return average across all samples
        return np.mean(fn_samples, axis=0)
    else: 
        return fn_samples


def mv_block_monte_carlo(fn, mu_arr, var_arr, fn_args =[], generator=None, num_samples=100, average=True):
    """
    multi-variate monte-carlo 

    Args:
        fn: Callable - 
        mu_arr: N x Q x B
        var_arr: N x 1 x QB x QB
    """

    if generator == None: raise RuntimeError()
    chex.assert_rank([mu_arr, var_arr], [3, 4])
    chex.assert_equal(var_arr.shape[1], 1)

    var_arr = var_arr[:, 0, :, :]

    N, Q, B = mu_arr.shape

    chex.assert_equal(var_arr.shape, (N, Q * B, Q*B))

    # TODO: test reshaping of mu_arr
    mu_arr = np.reshape(mu_arr, [N, Q * B])

    white_samples = objax.random.normal([num_samples]+list(mu_arr.shape), mean=0.0, stddev=1.0, generator=generator)
    chex.assert_shape(white_samples, [num_samples, N, Q*B])

    if True:
        # add jit for numerical stability when computing samples
        tiled_jit = np.tile(
            np.eye(var_arr.shape[1])*settings.jitter,
            [var_arr.shape[0], 1, 1]
        )
        chex.assert_equal_shape([var_arr, tiled_jit])
        chol_arr = np.linalg.cholesky(var_arr+tiled_jit)
    else:
        chol_arr = np.linalg.cholesky(var_arr)


    def reparameterise(fn, samples, mu_arr, chol_arr, *args):
        chex.assert_equal(samples.shape, mu_arr.shape)
        samples = samples[..., None]
        mu_arr = mu_arr[..., None]

        # reparemeterise
        s = mu_arr + chol_arr @ samples 

        # TODO: fix blocks here
        return fn(s, *args)


    num_args = len(fn_args)

    # batch over samples
    fn_samples = jax.vmap(
        reparameterise,
        [None, 0, None, None] + [None]*num_args,
        0
    )(fn, white_samples, mu_arr, chol_arr, *fn_args)

    if average:
        # Return average across all samples
        if type(fn_samples) is list:
            return [np.mean(f, axis=0) for f in fn_samples]
        return np.mean(fn_samples, axis=0)
    else: 
        return fn_samples



def mv_mean_field_block_monte_carlo(fn, mu_arr, var_arr, fn_args =[], generator=None, num_samples=100, average=True):
    """
    multi-variate monte-carlo

    Computes blocked samples across Q

    Args:
        fn: Callable - 
        mu_arr: N x Q x B
        var_arr: N x Q x B x B


    """

    if generator == None: raise RuntimeError()
    chex.assert_rank([mu_arr, var_arr], [3, 4])

    N, Q, B = mu_arr.shape

    chex.assert_equal(var_arr.shape, (N, Q, B, B))

    white_samples = objax.random.normal([num_samples]+list(mu_arr.shape), mean=0.0, stddev=1.0, generator=generator)
    chex.assert_shape(white_samples, [num_samples, N, Q, B])


    if True:
        # add jit for numerical stability when computing samples
        tiled_jit = np.tile(
            np.eye(B)*settings.jitter,
            [N, Q, 1, 1]
        )
        chol_arr = np.linalg.cholesky(var_arr+tiled_jit)
    else:
        chol_arr = np.linalg.cholesky(var_arr)


    def reparameterise(fn, samples, mu_arr, chol_arr, *args):
        chex.assert_equal(mu_arr.shape, (N, Q, B))
        chex.assert_equal(chol_arr.shape, (N, Q, B, B))
        chex.assert_equal_shape([samples, mu_arr])

        samples = samples[..., None]
        mu_arr = mu_arr[..., None]

        # reparemeterise
        s = mu_arr + chol_arr @ samples 

        return fn(s, *args)


    num_args = len(fn_args)

    # batch over samples and Q
    fn_samples = jax.vmap(
        reparameterise,
        [None, 0, None, None] + [None]*num_args,
        0
    )(fn, white_samples, mu_arr, chol_arr, *fn_args)

    if average:
        # Return average across all samples
        return np.mean(fn_samples, axis=0)
    else: 
        return fn_samples


def mv_block_monte_carlo_list(fn, mu_arr, var_arr, generator=None, num_samples=100, average=True):
    """
    multi-variate monte-carlo 

    Args:
        fn: Callable - 
        mu_arr: List[N x Q x B]
        var_arr: List[N x 1 x QB x QB]
    """

    if generator == None: raise RuntimeError()
    list_dim = len(mu_arr)

    var_arr = [V[:, 0, :, :] for V in var_arr]

    # reshape to 2d for sampling
    mu_arr = [np.reshape(mu, [mu.shape[0], -1]) for mu in mu_arr]

    white_samples = [
        objax.random.normal([num_samples]+list(mu.shape), mean=0.0, stddev=1.0, generator=generator)
        for mu in mu_arr
    ]
    
    def add_tiled_jit(V):
        tiled_jit = np.tile(
            np.eye(V.shape[1])*settings.jitter,
            [V.shape[0], 1, 1]
        )

        return V+ tiled_jit

    def get_chol_list(V):
        # add jit for numerical stability when computing samples
        chol_arr = [
            np.linalg.cholesky(add_tiled_jit(V)) for V in var_arr
        ]
        return chol_arr 

    chol_arr = get_chol_list(var_arr)

    def reparameterise(fn, mu_arr, chol_arr, *samples):
        samples = [s[..., None] for s in samples]
        mu_arr = [mu[..., None] for mu in mu_arr]

        # reparemeterise
        s = [mu_arr[i] + chol_arr[i] @ samples[i] for i in range(list_dim)]

        return fn(s)


    vmap_params = [fn, mu_arr, chol_arr] + white_samples
    # batch over samples
    fn_samples = jax.vmap(
        reparameterise,
        [None, None, None] + [0]*list_dim,
        0
    )(*vmap_params)

    if average:
        # Return average across all samples
        if type(fn_samples) is list:
            return [np.mean(f, axis=0) for f in fn_samples]

        return np.mean(fn_samples, axis=0)
    else: 
        return fn_samples
