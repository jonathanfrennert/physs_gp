"""
See http://proceedings.mlr.press/v33/solin14.pdf
"""
import objax
import chex
import jax
import jax.numpy as np

from . import Kernel, MarkovKernel

from .. import Parameter

from ..computation.custom import custom_bessel_ive

from tensorflow_probability.substrates import jax as tfp
from jax.scipy.linalg import expm

class __PeriodicBase(MarkovKernel):
    """ Two dimensional oscillatory SDE model """

    def __init__(self, frequency_param, lengthscale_param, variance_param, j, include_dt= False):
        self.j = j

        self.frequency_param = frequency_param
        self.lengthscale_param = lengthscale_param
        self.variance_param = variance_param

        self.include_dt = include_dt


    def to_ss(self, X_spatial=None):
        freq = self.frequency_param.value
        lengthscale = self.lengthscale_param.value
        variance= self.variance_param.value

        j = self.j

        F = np.array([
            [0, 1.0],
            [-freq * j, 0],
        ])

        L = np.eye(2)

        L = np.array([
            [1, 0],
            [0, 1]
        ])
        inv_ls = 1/(lengthscale**2)


        qj = freq*j*variance * tfp.math.bessel_ive(j, lengthscale) 

        if j != 0:
            qj = 2 * qj
        
        Qc = np.eye(2) * qj

        if not self.include_dt:
            H = np.array([
                [1.0, 0.0],
            ])

        else:
            H = np.array([
                [1.0, 0.0],
                [0.0, 1.0],
            ])


        Pinf = qj * np.eye(2)

        return F, L, Qc, H, Pinf



class _PeriodicBase(MarkovKernel):
    """ Two dimensional oscillatory SDE model """

    def __init__(self, frequency_param, lengthscale_param, variance_param, j, include_dt=False, include_dt2=False, use_custom_bessel_ive=False):
        self.j = j

        self.frequency_param = frequency_param
        self.lengthscale_param = lengthscale_param
        self.variance_param = variance_param
        self.include_dt = include_dt
        self.include_dt2 = include_dt2
        self.use_custom_bessel_ive = use_custom_bessel_ive


    def to_ss(self, X_spatial=None):
        freq = self.frequency_param.value
        lengthscale = self.lengthscale_param.value
        variance= self.variance_param.value

        j = self.j

        F = np.array([
            [0, - freq * j],
            [freq * j, 0],
        ])

        L = np.eye(2)

        # no noise in the derivative

        inv_ls = 1/(lengthscale**2)

        #L = np.array([
        #    [0, 0],
        #    [0, 1]
        #])

        L = np.array([
            [1, 0],
            [0, 1]
        ])


        # bessel_ive is an exponentially scaled version of the modified Bessel function of the first kind
        if self.use_custom_bessel_ive:
            #qj = variance * custom_bessel_ive(j, lengthscale, 1) 
            qj = variance * custom_bessel_ive(j, lengthscale**(-2), 1) 
        else:
            #qj = variance * tfp.math.bessel_ive(j, lengthscale) 
            qj = variance * tfp.math.bessel_ive(j, lengthscale**(-2)) 

        if j != 0:
            qj = 2 * qj
        
        #Qc = np.eye(2) * qj
        Qc = np.zeros(2)

        if self.include_dt:
            H = np.array([
                [1.0, 0.0],
                [0.0, -j * freq],
            ])
        elif self.include_dt2:
            H = np.array([
                [1.0, 0.0],
                [0.0, -j * freq],
                [-j * freq, 0.0],
            ])
        else:
            H = np.array([
                [1.0, 0.0],
            ])



        Pinf = qj * np.eye(2)

        return F, L, Qc, H, Pinf

class Periodic(Kernel):
    def __init__(self, frequency, lengthscale, variance, active_dims = None):
        super(Periodic, self).__init__(input_dim=1, active_dims=active_dims)

        self.lengthscale_param = Parameter(lengthscale, constraint='positive', name='Periodic/lengthscale')
        self.variance_param = Parameter(variance, constraint='positive', name='Periodic/variance')
        self.frequency_param = Parameter(frequency, constraint='positive', name='Periodic/frequency')

    def K_diag(self, X1):
        variance = self.variance_param.value
        return variance * np.ones(X1.shape[0])

    def _K_scaler(self, x1, x2):
        """
        Computes: tbd
        """
        ls = self.lengthscale_param.value
        variance = self.variance_param.value
        frequency = self.frequency_param.value

        tau = np.abs(x1 - x2)

        k = variance * np.exp(
            - 2 * np.square(
                np.sin( frequency * tau / 2) / ls
            )
        )

        return k

class ApproxSDEPeriodic_BN(MarkovKernel, Periodic):
    """ Adapted From https://github.com/AaltoML/BayesNewton/blob/main/bayesnewton/kernels.py#L804  and https://github.com/gpstuff-dev/gpstuff/blob/develop/gp/cf_periodic_to_ss.m"""

    def __init__(self, frequency, lengthscale, variance, n_terms=10, active_dims = None, include_dt=False, include_dt2=False, use_custom_bessel_ive = False):
        super(ApproxSDEPeriodic_BN, self).__init__(frequency, lengthscale, variance, active_dims=active_dims)

        self.n_terms = n_terms
        self.order = n_terms
        self.include_dt = include_dt
        self.include_dt2 = include_dt2
        self.use_custom_bessel_ive = use_custom_bessel_ive
        self.custom_bessel_interp_type = 1 #cubic

        self.base_kernel = None
        self._state_space_dim = self.state_size()

    @property
    def lengthscale(self):
        return  self.lengthscale_param.value

    @property
    def variance(self):
        return  self.variance_param.value

    @property
    def frequency(self):
        return  self.frequency_param.value

    def state_size(self):
        return int(2 * (self.n_terms+1))

    def to_ss(self, X_spatial=None):
        if self.use_custom_bessel_ive:
            bessel_fn = lambda A, b : np.array([custom_bessel_ive(a, b, self.custom_bessel_interp_type) for a in A])
        else:
            bessel_fn = tfp.math.bessel_ive

        q2 = np.array([1, *[2]*self.order]) * self.variance * bessel_fn([*range(self.order+1)], self.lengthscale**(-2))
        # The angular frequency
        omega = self.frequency

        # The model
        F = np.kron(np.diag(np.arange(self.order + 1)), np.array([[0., -omega], [omega, 0.]]))
        L = np.eye(2 * (self.order + 1))
        Qc = np.zeros(2 * (self.order + 1))
        Pinf = np.kron(np.diag(q2), np.eye(2))

        H_obs = np.kron(np.ones([1, self.order + 1]), np.array([1., 0.]))

        if self.include_dt2:
            H_diff = np.kron(-np.arange(self.order+1)*omega, np.array([0, 1]))
            H_diff_2 = np.kron(-np.arange(self.order+1)*omega, np.array([1, 0]))
            H  =  np.vstack([H_obs, H_diff, H_diff_2])
            
        elif self.include_dt:
            H_diff = np.kron(-np.arange(self.order+1)*omega, np.array([0, 1]))
            H  =  np.vstack([H_obs, H_diff])

        else:
            H = H_obs

        return F, L, Qc, H, Pinf


    def expm(self, dt, X_spatial=None):
        F, _, _, _, _ = self.to_ss(X_spatial) 

        return expm( F * dt)

class ApproxSDEPeriodic(MarkovKernel, Periodic):
    """ See TBD. """
    def __init__(self, frequency, lengthscale, variance, n_terms=10, active_dims = None, include_dt=False, include_dt2=False, use_custom_bessel_ive = False):

        super(ApproxSDEPeriodic, self).__init__(frequency, lengthscale, variance, active_dims=active_dims)

        self.n_terms = n_terms
        self.include_dt = include_dt
        self.include_dt2 = include_dt2
        self.use_custom_bessel_ive = use_custom_bessel_ive
        self.custom_bessel_interp_typ = 1 # cubic

        self.base_kernel = None
        self._setup_base_kernel()
        self._state_space_dim = self.state_size()


    def _setup_base_kernel(self):
        # order = n_terms + 1
        for n in range(self.n_terms+1):
            new_term = _PeriodicBase(
                self.frequency_param,
                self.lengthscale_param,
                self.variance_param,
                n,
                include_dt = self.include_dt,
                include_dt2 = self.include_dt2,
                use_custom_bessel_ive = self.use_custom_bessel_ive
            )

            if self.base_kernel == None:
                self.base_kernel = new_term
            else:
                self.base_kernel = self.base_kernel + new_term

    def state_size(self):
        return int(2 * (self.n_terms+1))

    def to_ss(self, X_spatial=None):
        return self.base_kernel.to_ss(X_spatial)


    def expm(self, dt, X_spatial=None):
        """appproximate matrix exponential A = expm(F * dt)"""

        F, _, _, _, _ = self.base_kernel.to_ss(X_spatial) 

        return expm( F * dt)





