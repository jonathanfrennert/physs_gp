import jax
import jax.numpy as np

from scipy.cluster.vq import kmeans2

from ..sparsity import NoSparsity, FullSparsity
from ..models import GP
from ..transforms.multi_output import LMC, LMC_DRD, GPRN, GPRN_Exp, GPRN_DRD, GPRN_DRD_EXP, Independent
from ..data import Data, TransformedData
from ..likelihood import Gaussian
from ..kernels import RBF, ScaleKernel
from stgp.approximate_posteriors import FullGaussianApproximatePosterior
from stgp.transforms.basic import Log, Softminus, Affine, ReverseFlow

def gp_regression(X, Y, P=None,  kernels = None, inference = 'Batch', lengthscale=1.0, variance=1.0, lik_noise = 0.1, normalise_data: bool = False, M=None, additive=False, whiten=False):
    """
    Helper function for returning an LMC model with Gaussian likelihood across all outputs.

    Args:
        P, Q: When not passed they are set to Y.shape[1]
        kernels: When not passed they are set to RBF kernels
    """

    D = X.shape[1]

    # set defaults
    if P is None:
        P = Y.shape[1]

    Q = P

    if kernels is None:
        kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*lengthscale, additive = additive),
                variance = variance
            )
            for q in range(Q)
        ]


    # construct model
    if M is None:
        Z = [NoSparsity(X) for q in range(Q)]
        M = X.shape[0]
    else:
        Z_centroid, _ = kmeans2(X, M, seed=0)
        Z = [FullSparsity(Z_centroid) for q in range(Q)]


    latent_gps = [
        GP(sparsity=Z[q], kernel=kernels[q]) for q in range(Q)
    ] 

    # Construct LMC Prior
    prior = Independent(latent_gps)

    if inference == 'vi':
        inference='Variational'
        approximate_posterior = None
    else:
        approximate_posterior = None



    if normalise_data:
        data = TransformedData(
            Data(X, Y), 
            [
                ReverseFlow(Affine(np.nanstd(Y[:, i]), np.nanmean(Y[:, i]), train=False)) for i in range(Y.shape[1])
            ]
        )
    else:
        data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=lik_noise) for p in range(P)],
        prior=prior,
        inference=inference,
        approximate_posterior = approximate_posterior,
        whiten = whiten
    )

    return m


def lmc_regression(X, Y, P=None, Q=None, kernels = None, inference = 'Batch', lengthscale=1.0, variance=1.0, lik_noise = 0.1, normalise_data: bool = False, M=None, additive=False, whiten=False):
    """
    Helper function for returning an LMC model with Gaussian likelihood across all outputs.

    Args:
        P, Q: When not passed they are set to Y.shape[1]
        kernels: When not passed they are set to RBF kernels
    """

    D = X.shape[1]

    # set defaults
    if P is None:
        P = Y.shape[1]

    if Q is None:
        Q = P

    if kernels is None:
        kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*lengthscale, additive = additive),
                variance = variance
            )
            for q in range(Q)
        ]


    # construct model
    if M is None:
        Z = [NoSparsity(X) for q in range(Q)]
        M = X.shape[0]
    else:
        Z_centroid, _ = kmeans2(X, M, seed=0)
        Z = [FullSparsity(Z_centroid) for q in range(Q)]


    latent_gps = [
        GP(sparsity=Z[q], kernel=kernels[q]) for q in range(Q)
    ] 

    # Construct LMC Prior
    prior = LMC(latent_gps, output_dim = P)

    if inference == 'vi':
        inference='Variational'
        approximate_posterior = FullGaussianApproximatePosterior(
            dim = M*prior.base_prior.output_dim
        ) 
    else:
        approximate_posterior = None



    if normalise_data:
        data = TransformedData(
            Data(X, Y), 
            [
                ReverseFlow(Affine(np.nanstd(Y[:, i]), np.nanmean(Y[:, i]), train=False)) for i in range(Y.shape[1])
            ]
        )
    else:
        data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=lik_noise) for p in range(P)],
        prior=prior,
        inference=inference,
        approximate_posterior = approximate_posterior,
        whiten = whiten
    )

    return m

def lmc_drd_regression(X, Y, P=None, Q=None, kernels = None, inference = 'Batch', lengthscale=1.0, variance=1.0, lik_noise = 0.1, normalise_data: bool = False, M=None, additive=False, whiten=False):
    """
    Helper function for returning an LMC-DRD model with Gaussian likelihood across all outputs.

    Args:
        P, Q: When not passed they are set to Y.shape[1]
        kernels: When not passed they are set to RBF kernels
    """

    D = X.shape[1]

    # set defaults
    if P is None:
        P = Y.shape[1]

    if Q is None:
        Q = P

    if kernels is None:
        # DRD should not have a scale kernel 
        kernels = [
            RBF(input_dim = D, lengthscales=np.ones(D)*lengthscale, additive = additive)
            for q in range(Q)
        ]

    # construct model
    if M is None:
        Z = [NoSparsity(X) for q in range(Q)]
        M = X.shape[0]
    else:
        Z_centroid, _ = kmeans2(X, M, seed=0)
        Z = [FullSparsity(Z_centroid) for q in range(Q)]

    latent_gps = [
        GP(sparsity=Z[q], kernel=kernels[q]) for q in range(Q)
    ] 

    prior = LMC_DRD(latent_gps, output_dim = P)

    if inference == 'vi':
        inference='Variational'
        approximate_posterior = FullGaussianApproximatePosterior(
            dim = M*prior.base_prior.output_dim
        ) 
    else:
        approximate_posterior = None


    if normalise_data:
        data = TransformedData(
            Data(X, Y), 
            [
                ReverseFlow(Affine(np.nanstd(Y[:, i]), np.nanmean(Y[:, i]), train=False)) for i in range(Y.shape[1])
            ]
        )
    else:
        data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=lik_noise) for p in range(P)],
        prior=prior,
        inference=inference,
        approximate_posterior = approximate_posterior,
        whiten = whiten
    )

    return m


def gprn_regression(X, Y, P=None, Q=None, W_kernels=None, f_kernels=None, inference='Variational', constraint=None, ell_samples=100, lengthscale=1.0, W_lengthscales = None, f_lengthscales = None, variance=1.0, lik_noise=0.1, normalise_data=True, M=None, additive=False, whiten=False):
    """ Helper function for returning a variational mean-field GPRN model with Gaussian likelihood across all outputs.  """

    D = X.shape[1]

    # set defaults
    if P is None:
        P = Y.shape[1]

    if Q is None:
        Q = P

    if W_lengthscales is None:
        W_lengthscales = [
            [
                lengthscale
                for q in range(Q)
            ]
            for p in range(P)
        ]

    if f_lengthscales is None:
        f_lengthscales = [lengthscale for q in range(Q)]

    W_lengthscales = np.array(W_lengthscales)
    f_lengthscales = np.array(f_lengthscales)

    if W_kernels is None:
        W_kernels = [
            [
                ScaleKernel(
                    RBF(input_dim = D, lengthscales=np.ones(D)*W_lengthscales[p, q], additive = additive),
                    variance = variance
                )
                for q in range(Q)
            ]
            for p in range(P)
        ]

    if f_kernels is None:
        f_kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*f_lengthscales[q], additive = additive),
                variance = variance
            )
            for q in range(Q)
        ]

    if inference == 'vi':
        inference='Variational'

    # setup model

    Z_f = [NoSparsity(X) for q in range(Q)]
    Z_W = [[NoSparsity(X) for q in range(Q)] for p in range(P)]


    # construct model
    if M is None:
        Z_f = [NoSparsity(X) for q in range(Q)]
        Z_W = [[NoSparsity(X) for q in range(Q)] for p in range(P)]
        M = X.shape[0]
    else:
        Z_centroid, _ = kmeans2(X, M, seed=0)
        Z_f = [FullSparsity(Z_centroid) for q in range(Q)]
        Z_W = [[FullSparsity(Z_centroid) for q in range(Q)] for p in range(P)]

    latent_f_gps = [
        GP(sparsity=Z_f[q], kernel=f_kernels[q]) for q in range(Q)
    ] 

    latent_W_gps = [
        [GP(sparsity=Z_W[p][q], kernel=W_kernels[p][q]) for q in range(Q)]
        for p in range(P)
    ] 

    # Construct LMC Prior
    if constraint is None:
        prior = GPRN(latent_W_gps, latent_f_gps, output_dim = P)
    elif constraint == 'exp':
        prior = GPRN_Exp(latent_W_gps, latent_f_gps, output_dim = P)
    else:
        raise NotImplementedError()

    if normalise_data:
        data = TransformedData(
            Data(X, Y), 
            [
                ReverseFlow(Affine(np.nanstd(Y[:, i]), np.nanmean(Y[:, i]), train=False)) for i in range(Y.shape[1])
            ]
        )
    else:
        data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=lik_noise) for p in range(P)],
        prior=prior,
        inference=inference,
        ell_samples=ell_samples,
        prediction_samples=None,
        whiten = whiten
    )

    return m


def gprn_drd_regression(X, Y, P=None, W_kernels=None, f_kernels=None, latent_variance = 1.0, variance = 1.0, ell_samples=100, lengthscale=1.0,  W_lengthscales=None, f_lengthscales=None, lik_noise=0.1, meanfield=True, normalise_data=True, M=None, additive=False, whiten=False):
    """ Helper function for returning a variational full-Gaussian GPRN_DRD model with Gaussian likelihood across all outputs.  """

    D = X.shape[1]

    # set defaults
    if P is None:
        P = Y.shape[1]

    Q = P

    num_W = int(Q * (Q-1)/2)

    num_latents = Q + num_W

    if W_lengthscales is None:
        W_lengthscales = [lengthscale for i in range(num_W)]

    if f_lengthscales is None:
        f_lengthscales = [lengthscale for i in range(Q)]

    if W_kernels is None:
        W_kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*W_lengthscales[q], additive = additive),
                variance = variance
            )
            for q in range(num_W)
        ]

    if f_kernels is None:
        # kernel variances must be 1, so that K is a correlation matrix
        if additive:
            scale_var = 1/D
        else:
            scale_var = 1

        f_kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*f_lengthscales[q], additive = additive),
                variance = scale_var
            )
            for q in range(Q)
        ]

        # fix f kernels variance
        [kern.variance_param.fix() for kern in f_kernels]

    # setup model

    if M is None:
        Z_f = [NoSparsity(X) for q in range(Q)]
        Z_W = [NoSparsity(X) for q in range(num_W)]
        M = X.shape[0]
    else:
        Z_centroid, _ = kmeans2(X, M, seed=0)
        Z_f = [FullSparsity(Z_centroid) for q in range(Q)]
        Z_W = [FullSparsity(Z_centroid) for q in range(num_W)]

    latent_f_gps = [
        GP(sparsity=Z_f[q], kernel=f_kernels[q]) for q in range(Q)
    ] 

    latent_W_gps = [
        GP(sparsity=Z_W[q], kernel=W_kernels[q]) for q in range(num_W)
    ] 


    prior = GPRN_DRD(
        latent_W_gps, 
        latent_f_gps,
        input_dim = Q,
        output_dim = P,
        variances = np.ones(P)*latent_variance
    )

    if meanfield:
        q = None
    else:
        q = FullGaussianApproximatePosterior(
            dim = M*prior.base_prior.output_dim
        )

    if normalise_data:
        data = TransformedData(
            Data(X, Y), 
            [
                ReverseFlow(Affine(np.nanstd(Y[:, i]), np.nanmean(Y[:, i]), train=False)) for i in range(Y.shape[1])
            ]
        )
    else:
        data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=lik_noise) for p in range(P)],
        prior=prior,
        inference='Variational',
        approximate_posterior = q,
        ell_samples=ell_samples,
        prediction_samples=None,
        whiten = whiten
    )

    return m


def gprn_drd_nv_regression(X, Y, P=None, W_kernels=None, f_kernels=None, v_kernels=None, latent_variance = 1.0, variance = 1.0, ell_samples=100, lengthscale=1.0,  v_lengthscales = None, W_lenthscales=None, f_lengthscales=None, lik_noise=0.1, meanfield=True, normalise_data=True, M=None, additive=False, whiten=False):
    """ Helper function for returning a variational full-Gaussian Noise Varying GPRN_DRD model with Gaussian likelihood across all outputs.  """

    D = X.shape[1]

    # set defaults
    if P is None:
        P = Y.shape[1]

    Q = P

    num_W = int(Q * (Q-1)/2)

    if v_lengthscales is None:
        v_lengthscales = np.ones(P)*lengthscale
    if W_lenthscales is None:
        W_lenthscales = np.ones(num_W)*lengthscale
    if f_lengthscales is None:
        f_lengthscales = np.ones(Q)*lengthscale

    if v_kernels is None:
        v_kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*v_lengthscales[q], additive = additive),
                variance = variance
            )
            for q in range(P)
        ]

    if W_kernels is None:
        W_kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*W_lenthscales[q], additive = additive),
                variance = variance
            )
            for q in range(num_W)
        ]

    if f_kernels is None:
        # kernel variances must be 1, so that K is a correlation matrix
        f_kernels = [
            ScaleKernel(
                RBF(input_dim = D, lengthscales=np.ones(D)*f_lengthscales[q], additive = additive),
                variance = variance
            )
            for q in range(Q)
        ]

        # fix f kernels variance to ensure a correlation matrix
        [kern.variance_param.fix() for kern in f_kernels]

    # setup model

    Z_v = [NoSparsity(X) for p in range(P)]
    Z_f = [NoSparsity(X) for q in range(Q)]
    Z_W = [NoSparsity(X) for q in range(num_W)]

    if M is None:
        Z_v = [NoSparsity(X) for p in range(P)]
        Z_f = [NoSparsity(X) for q in range(Q)]
        Z_W = [NoSparsity(X) for q in range(num_W)]
        M = X.shape[0]
    else:
        Z_centroid, _ = kmeans2(X, M, seed=0)

        Z_v = [FullSparsity(Z_centroid) for p in range(P)]
        Z_f = [FullSparsity(Z_centroid) for q in range(Q)]
        Z_W = [FullSparsity(Z_centroid) for q in range(num_W)]



    latent_v_gps = [
        GP(sparsity=Z_v[q], kernel=v_kernels[q]) for q in range(P)
    ] 

    latent_f_gps = [
        GP(sparsity=Z_f[q], kernel=f_kernels[q]) for q in range(Q)
    ] 

    latent_W_gps = [
        GP(sparsity=Z_W[q], kernel=W_kernels[q]) for q in range(num_W)
    ] 


    prior = GPRN_DRD_EXP(
        latent_v_gps, 
        latent_W_gps, 
        latent_f_gps,
        input_dim = Q,
        output_dim = P,
        variances = np.ones(P)*latent_variance
    )

    if meanfield:
        q = None
    else:
        q = FullGaussianApproximatePosterior(
            dim = M*prior.base_prior.output_dim
        )

    if normalise_data:
        data = TransformedData(
            Data(X, Y), 
            [
                ReverseFlow(Affine(np.nanstd(Y[:, i]), np.nanmean(Y[:, i]), train=False)) for i in range(Y.shape[1])
            ]
        )
    else:
        data = Data(X, Y)

    m = GP(
        data=data,
        likelihood = [Gaussian(variance=lik_noise) for p in range(P)],
        prior=prior,
        inference='Variational',
        approximate_posterior = q,
        ell_samples=ell_samples,
        prediction_samples=None,
        whiten = whiten
    )

    return m
