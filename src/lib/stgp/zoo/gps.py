import jax
import stgp
from stgp.models import GP
from stgp.data import Data, SpatioTemporalData, TemporallyGroupedData, get_sequential_data_obj
from stgp.transforms import Independent
from stgp.likelihood import ProductLikelihood, ReshapedGaussian
from stgp.likelihood.product_likelihood import get_product_likelihood
from stgp.transforms.sdes import LTI_SDE
from stgp.approximate_posteriors import FullConjugateGaussian
from stgp.sparsity import NoSparsity, SpatialSparsity, StackedSparsity, StackedNoSparsity, FullSparsity
from stgp.kernels import SpatioTemporalSeperableKernel, Matern32, RBF
from stgp.likelihood import Gaussian
import numpy as np

def sde_gp(X, Y, kernel, likelihood, seed=0, filter_type='sequential'):
    data = get_sequential_data_obj(X, Y, sort=True)

    lik = ReshapedGaussian(likelihood, num_blocks=data.Nt, block_size=data.Ns)

    m = GP(
        data = data,
        prior = LTI_SDE(Independent([
            GP(
                sparsity = stgp.sparsity.NoSparsity(X), 
                kernel = kernel,
                prior = True
            )
        ])),
        likelihood = lik,
        inference='Sequential',
        filter_type=filter_type
    )

    return m


def batch_gp(X, Y, kernel, likelihood, seed=0):
    data = Data(X, Y)

    m = GP(
        data = data,
        prior = Independent([
            GP(
                sparsity = stgp.sparsity.NoSparsity(X), 
                kernel = kernel,
                prior = True
            )
        ]),
        likelihood = get_product_likelihood([likelihood])
    )

    return m

def vgp(X, Y,  kernel, likelihood, minibatch_size=None, seed=0):

    if minibatch_size is not None:
        data = Data(X, Y, minibatch_size = minibatch_size, seed=0)
    else:
        data = Data(X, Y)

    m = GP(
        data = data,
        prior = Independent([
            GP(
                sparsity = stgp.sparsity.NoSparsity(X), 
                kernel = kernel,
                prior = True
            )
        ]),
        likelihood = ProductLikelihood([likelihood]),
        inference='Variational'
    )

    return m


def svgp(X, Y, Z, kernel, likelihood, minibatch_size=None, seed=0):
    if minibatch_size is not None:
        data = Data(X, Y, minibatch_size = minibatch_size, seed=seed)
    else:
        data = Data(X, Y)

    m = GP(
        data = data,
        prior = Independent([
            GP(
                sparsity = stgp.sparsity.FullSparsity(Z = Z), 
                kernel = kernel,
                prior = True
            )
        ]),
        likelihood = ProductLikelihood([likelihood]),
        inference='Variational'
    )

    return m


def stvgp(X, Y, Zs, kernel, likelihood, minibatch_size=None, seed=0, parallel=False):
    """
    A Spatio-temporal variational Gaussian Process 

    This is a spatially sparse variational Gaussian processes that exploits natural gradients to represent the approximate posterior
        as a conjugate spatio-temporal state-space model
    """
    if minibatch_size:
        # TODO: add this
        raise NotImplementedError('spatial minibatching is not currently supported using this zoo methods')

    Q = 1
    P = 1
    st_data = TemporallyGroupedData(X=X, Y=Y)

    Z = [SpatialSparsity(st_data.X_time, Zs, train=True) for q in range(Q)]

    Z_all = StackedSparsity(Z)

    # Construct Latent GPs
    latent_kernels = [
        kernel
        for q in range(Q)
    ]
    latent_gps = [
        stgp.models.GP(sparsity=Z[0], kernel=latent_kernels[q]) for q in range(Q)
    ] 
    np.random.seed(0)

    prior = stgp.transforms.Independent(latent_gps)

    # Construct Full Gaussian Approximate Posterior
    Mt = Z[0].raw_Z.Nt
    Ms = Z[0].raw_Z.Ns


    q = FullConjugateGaussian(
        X = Z[0], # for state-space models we require the same Z across all latents
        num_latents=Q,
        block_size=Q*Z[0].raw_Z.Ns,
        num_blocks = st_data.Nt,
        surrogate_model = lambda X, Y, likelihood:  stgp.models.GP(
            data = SpatioTemporalData(X=X.raw_Z, Y=np.reshape(Y, [Mt, Q, Ms]), sort=False), # we need gradients Y so set to be trainable, in time-latent-space format
            likelihood=likelihood, 
            prior=LTI_SDE(Independent(latent_gps)),
            inference='Sequential',
            full_state_observed = False,
            parallel=parallel
        )
    )

    m = stgp.models.GP(
        data=st_data, 
        likelihood=[likelihood for p in range(P)],
        inference='Variational',
        prior=prior,
        approximate_posterior = q,
        whiten=False
    )

    return m
