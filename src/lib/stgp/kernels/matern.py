import objax
import chex
import jax
import jax.numpy as np

from . import StationaryKernel, StationaryVarianceKernel, MarkovKernel
from .ss_utils import matern32_temporal_expm, matern32_temporal_state_space_rep


class ScaledMatern32(StationaryVarianceKernel, MarkovKernel):
    def state_space_dim(self):
        return 2

    def to_ss(self, X_spatial=None):
        """ Return state space representation """
        chex.assert_equal(self.input_dim, 1)

        lengthscale = self.lengthscales[0]

        return matern32_temporal_state_space_rep(
            self.lengthscales[0],
            self.variance
        )


    def state_size(self):
        return 2

    def expm(self, dt, X_spatial=None):
        """closed form matrix exponential A = expm(F * dt)"""
        chex.assert_equal(self.input_dim, 1)

        return matern32_temporal_expm(
            dt,
            self.lengthscales[0]
        )


    def _K_scaler_with_var(self, x1, x2, lengthscale, variance):
        """
        K(X1, X2) = σ (1 + √3 (X1-X2)/l) exp{-√3 (X1-X2)/l}
        """

        r  = np.abs(x1-x2) / lengthscale
        sqrt3 = np.sqrt(3.0)

        return  variance * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)


class Matern32(StationaryKernel, MarkovKernel):
    def __init__(self, *args, **kwargs):
        super(Matern32, self).__init__(*args, **kwargs, name='Matern32')

        self._state_space_dim = 2

    def to_ss(self, X_spatial=None):
        """ Return state space representation """
        chex.assert_equal(self.input_dim, 1)

        lengthscale = self.lengthscales[0]

        # no variance so just pass 1.0
        return matern32_temporal_state_space_rep(
            lengthscale, 1.0
        )

    def state_size(self):
        return 2

    def expm(self, dt, X_spatial=None):
        """closed form matrix exponential A = expm(F * dt)"""
        chex.assert_equal(self.input_dim, 1)

        return matern32_temporal_expm(
            dt, 
            self.lengthscales[0]
        )

    def _K_scaler(self, x1, x2, lengthscale):
        """
        K(X1, X2) = σ (1 + √3 (X1-X2)/l) exp{-√3 (X1-X2)/l}
        """

        r  = np.abs(x1-x2) / lengthscale
        sqrt3 = np.sqrt(3.0)

        return  (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)

class Matern12(StationaryKernel, MarkovKernel):
    def __init__(self, *args, **kwargs):
        super(Matern12, self).__init__(*args, **kwargs, name='Matern12')

        self._state_space_dim = 1

    def cf_to_ss_temporal(self):
        chex.assert_equal(self.input_dim, 1)
        raise NotImplementedError()

    def _K_scaler(self, x1, x2, lengthscale):
        """
        K(X1, X2) = σ²  exp{-|X1-X2|/l}
        """

        return np.exp(- np.abs(x1-x2) / lengthscale)

class Matern52(StationaryKernel, MarkovKernel):
    def __init__(self, *args, **kwargs):
        super(Matern52, self).__init__(*args, **kwargs, name='Matern52')
        self.variance = 1.0
        self._state_space_dim = 3

    def to_ss(self, X_spatial=None):
        """ Return state space representation """
        chex.assert_equal(self.input_dim, 1)

        var = self.variance
        ls = self.lengthscales[0]


        lam = 5.0**0.5 / ls
        F = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-(lam**3.0), -3.0 * lam**2.0, -3.0 * lam],
            ]
        )
        L = np.array([[0.0], [0.0], [1.0]])
        Qc = np.array(
            [[var * 400.0 * 5.0**0.5 / 3.0 / ls**5.0]]
        )
        H = np.array([[1.0, 0.0, 0.0]])
        kappa = 5.0 / 3.0 * var / ls**2.0

        minf = np.zeros([3, 1])
        Pinf = np.array(
            [
                [var, 0.0, -kappa],
                [0.0, kappa, 0.0],
                [-kappa, 0.0, 25.0 * var / ls**4.0],
            ]
        )
        return F, L, Qc, H, minf, Pinf

    def state_size(self):
        return 3


    def expm(self, dt, X_spatial=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-5/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [3, 3]
        """
        ls = self.lengthscales[0]

        lam = np.sqrt(5.0) / ls
        dtlam = dt * lam
        A = np.exp(-dtlam) * (
            dt
            * np.array(
                [
                    [lam * (0.5 * dtlam + 1.0), dtlam + 1.0, 0.5 * dt],
                    [-0.5 * dtlam * lam**2, lam * (1.0 - dtlam), 1.0 - 0.5 * dtlam],
                    [
                        lam**3 * (0.5 * dtlam - 1.0),
                        lam**2 * (dtlam - 3),
                        lam * (0.5 * dtlam - 2.0),
                    ],
                ]
            )
            + np.eye(3)
        )
        return A

    def _K_scaler(self, x1, x2, lengthscale):
        """
        r = |X1 - X2|/l
        K(X1, X2) = σ (1 + √5 r + (5/3) r^2) exp{-√5 r}
        """

        r  = np.abs(x1-x2) / lengthscale
        sqrt5 = np.sqrt(5.0)

        return  (1.0 + sqrt5 * r + (5.0/3.0) * r * r) * np.exp(-sqrt5 * r)


class ScaledMatern52(StationaryVarianceKernel, MarkovKernel):
    """ See https://github.com/AaltoML/BayesNewton/blob/main/bayesnewton/kernels.py """
    def state_space_dim(self):
        return 3

    def to_ss(self, X_spatial=None):
        """ Return state space representation """
        chex.assert_equal(self.input_dim, 1)

        var = self.variance
        ls = self.lengthscales[0]

        lam = 5.0**0.5 / ls
        F = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-(lam**3.0), -3.0 * lam**2.0, -3.0 * lam],
            ]
        )
        L = np.array([[0.0], [0.0], [1.0]])
        Qc = np.array(
            [[var * 400.0 * 5.0**0.5 / 3.0 / ls**5.0]]
        )
        H = np.array([[1.0, 0.0, 0.0]])
        kappa = 5.0 / 3.0 * var / ls**2.0
        minf = np.zeros([3, 1])
        Pinf = np.array(
            [
                [var, 0.0, -kappa],
                [0.0, kappa, 0.0],
                [-kappa, 0.0, 25.0 * var / ls**4.0],
            ]
        )
        return F, L, Qc, H, minf, Pinf

    def state_size(self):
        return 3

    def expm(self, dt, X_spatial=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-5/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [3, 3]
        """
        ls = self.lengthscales[0]

        lam = np.sqrt(5.0) / ls
        dtlam = dt * lam
        A = np.exp(-dtlam) * (
            dt
            * np.array(
                [
                    [lam * (0.5 * dtlam + 1.0), dtlam + 1.0, 0.5 * dt],
                    [-0.5 * dtlam * lam**2, lam * (1.0 - dtlam), 1.0 - 0.5 * dtlam],
                    [
                        lam**3 * (0.5 * dtlam - 1.0),
                        lam**2 * (dtlam - 3),
                        lam * (0.5 * dtlam - 2.0),
                    ],
                ]
            )
            + np.eye(3)
        )
        return A

    def _K_scaler_with_var(self, x1, x2, lengthscale, variance):
        """
        r = |X1 - X2|/l
        K(X1, X2) = σ (1 + √5 r + (5/3) r^2) exp{-√5 r}
        """

        r  = np.abs(x1-x2) / lengthscale
        sqrt5 = np.sqrt(5.0)

        return  variance * (1.0 + sqrt5 * r + (5.0/3.0) * r * r) * np.exp(-sqrt5 * r)


class ScaledMatern72(StationaryVarianceKernel, MarkovKernel):
    """ See https://github.com/AaltoML/BayesNewton/blob/main/bayesnewton/kernels.py#L280 """

    def state_space_dim(self):
        return 4

    def to_ss(self, X_spatial=None):
        """ Return state space representation """
        chex.assert_equal(self.input_dim, 1)

        var = self.variance
        ls = self.lengthscales[0]

        lam = 7.0**0.5 / ls
        F = np.array([[0.0,       1.0,           0.0,           0.0],
                      [0.0,       0.0,           1.0,           0.0],
                      [0.0,       0.0,           0.0,           1.0],
                      [-lam**4.0, -4.0*lam**3.0, -6.0*lam**2.0, -4.0*lam]])
        L = np.array([[0.0],
                      [0.0],
                      [0.0],
                      [1.0]])
        Qc = np.array([[var * 10976.0 * 7.0 ** 0.5 / 5.0 / ls ** 7.0]])
        H = np.array([[1, 0, 0, 0]])
        kappa = 7.0 / 5.0 * var / ls**2.0
        kappa2 = 9.8 * var / ls**4.0
        minf = np.zeros([4, 1])
        Pinf = np.array([[var,   0.0,    -kappa, 0.0],
                         [0.0,    kappa,   0.0,    -kappa2],
                         [-kappa, 0.0,     kappa2, 0.0],
                         [0.0,    -kappa2, 0.0,    343.0*var / ls**6.0]])

        return F, L, Qc, H, minf, Pinf

    def state_size(self):
        return 4

    def expm(self, dt, X_spatial=None):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-5/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [3, 3]
        """
        ls = self.lengthscales[0]

        lam = np.sqrt(7.0) / ls
        lam2 = lam * lam
        lam3 = lam2 * lam
        dtlam = dt * lam
        dtlam2 = dtlam ** 2
        A = np.exp(-dtlam) \
            * (dt * np.array([[lam * (1.0 + 0.5 * dtlam + dtlam2 / 6.0),      1.0 + dtlam + 0.5 * dtlam2,
                              0.5 * dt * (1.0 + dtlam),                       dt ** 2 / 6],
                              [-dtlam2 * lam ** 2.0 / 6.0,                    lam * (1.0 + 0.5 * dtlam - 0.5 * dtlam2),
                              1.0 + dtlam - 0.5 * dtlam2,                     dt * (0.5 - dtlam / 6.0)],
                              [lam3 * dtlam * (dtlam / 6.0 - 0.5),            dtlam * lam2 * (0.5 * dtlam - 2.0),
                              lam * (1.0 - 2.5 * dtlam + 0.5 * dtlam2),       1.0 - dtlam + dtlam2 / 6.0],
                              [lam2 ** 2 * (dtlam - 1.0 - dtlam2 / 6.0),      lam3 * (3.5 * dtlam - 4.0 - 0.5 * dtlam2),
                              lam2 * (4.0 * dtlam - 6.0 - 0.5 * dtlam2),      lam * (1.5 * dtlam - 3.0 - dtlam2 / 6.0)]])
               + np.eye(4))
        return A

    def _K_scaler_with_var(self, x1, x2, lengthscale, variance):
        """
        r = |X1 - X2|/l
        K(X1, X2) = σ (1 + √5 r + (5/3) r^2) exp{-√5 r}
        """

        r  = np.abs(x1-x2) / lengthscale

        sqrt7 = np.sqrt(7.0)
        return variance * (1. + sqrt7 * r + 14. / 5. * np.square(r) + 7. * sqrt7 / 15. * r**3) * np.exp(-sqrt7 * r)
