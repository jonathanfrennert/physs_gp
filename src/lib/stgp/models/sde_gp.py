import objax
import jax
import jax.numpy as np
import numpy as onp
import chex
from typing import Optional, Tuple
import warnings

from ..dispatch import dispatch
from ..dispatch import evoke
from ..core import Model, Posterior
from . import GP, BatchGP
from ..computation.filters import kalman_filter, rts_smoother, parallel_kalman_filter, parallel_rts_smoother, square_root_kalman_filter
from ..computation.matrix_ops import batched_block_diagional
from ..computation.permutations import permute_mat, permute_vec, permute_vec_ld_to_dl, permute_mat_ld_to_dl, permute_vec_tps_to_tsp, permute_mat_tps_to_tsp

from ..defaults import get_default_likelihood
from ..data import TemporalData, SpatioTemporalData, get_sequential_data_obj, SpatialTemporalInput, TemporallyGroupedData
from ..data.sequential import add_temporal_points
from ..kernels import Matern32
from ..likelihood import get_product_likelihood, ProductLikelihood
from ..transforms import Independent
from ..transforms.sdes import LTI_SDE, LTI_SDE_Full_State_Obs



from ..defaults import get_default_kernel, get_default_likelihood, get_default_independent_prior
from ..sparsity import NoSparsity

def get_R_R_inv(likelihood):
    """
    Note: that when predicting on xs the variance is always used
    """
    if str(type(likelihood).__name__) in ['BlockGaussianProductLikelihood']:
        likelihood = likelihood.likelihood_arr[0]

    if str(type(likelihood).__name__) in ['PrecisionBlockDiagonalGaussian']:
        return None, likelihood.precision

    elif str(type(likelihood).__name__) in ['ReshapedGaussian', 'ReshapedDiagonalGaussian', 'ReshapedBlockDiagonalGaussian', 'BlockDiagonalGaussian']:
        return likelihood.variance, None

    raise RuntimeError('Likelihood type not found!')

@dispatch(Model, 'Sequential')
class SDE_GP(Posterior):
    def __new__(cls, data, *args, **kwargs):
        if isinstance(data, SpatioTemporalData):
            return ST_SDE_GP(data, *args, **kwargs)
        else:
            return T_SDE_GP(data, *args, **kwargs)

class BASE_SDE_GP(Posterior):
    def __init__(
        self, 
        data = None,
        Z = None,
        inference: 'Sequential'=None, 
        likelihood: 'Likelihood'=None, 
        kernel: 'Kernel'=None, 
        prior: 'Transform'=None, 
        whiten=False, 
        fix_input=True,
        full_state_observed = False,
        filter_type = 'sequential',
        parallel = None,
        **kwargs
    ):
        # Use the sorted X and Y to construct the model on
        super(BASE_SDE_GP, self).__init__(data=data)

        if parallel is not None:
            raise DeprecationWarning('parllel is depreciated, use filter_type')

        self._likelihood = likelihood
        self.kernel = kernel
        self._prior = prior

        self.full_state_observed = full_state_observed

        self.set_defaults()

        self.filter_type  = filter_type

        

    @property
    def likelihood(self): return self._likelihood 

    @property
    def input_space_dim(self): return self.data.X.shape[1]

    @property
    def output_dim(self): 
        P = self.data.P

        if self.full_state_observed:
            return self.prior.output_dim

        return P

    @property
    def X(self): return self.data.X 

    @property
    def Y(self): return self.data.Y 

    @property
    def prior(self): return self._prior

    @property
    def Nt(self): 
        """ Return the number of temporal points. """
        return self.data.Nt 

    def set_defaults(self):
        """ Replace missing options with defaults """

        if (self.kernel is not None) and (self.prior is not None):
            raise RuntimeError('Only kernel or a prior must be passed')

        if self.prior is None:
            if self.kernel is None: 
                raise RuntimeError('Kernel must be passed!')

            if type(self.kernel) is not list:
                self.kernel = [self.kernel]

            # Pass object to avoid storing multiple copies of X
            X_ref = self.data._X
            sparsity = [
                NoSparsity(Z_ref = X_ref) 
                for q in range(self.output_dim)
            ]

            # Construct independent prior
            self._prior = get_default_independent_prior(
                sparsity,
                self.input_space_dim, 
                self.output_dim, 
                kernel_list=self.kernel
            )
            if self.full_state_observed:
                self._prior = LTI_SDE_Full_State_Obs(self._prior)
            else:
                self._prior = LTI_SDE(self._prior)

        if self.likelihood == None:
            self._likelihood = get_default_likelihood(self.output_dim)[0]

        if type(self.likelihood) == list:
            self._likelihood = get_product_likelihood(self._likelihood)


    def log_marginal_likelihood(self):
        R, R_inv = get_R_R_inv(self.likelihood)

        lml, _  = kalman_filter.filter_loop(
            self.data,
            self.prior,
            R = R,
            R_inv = R_inv,
            filter_type = self.filter_type
        )

        return lml

    def get_objective(self):
        return -self.log_marginal_likelihood()

    def mean(self, XS):
        mu, _ = self.predict_f(XS, diagonal=True, squeeze=False)
        return mu

    def var(self, XS):
        _, var = self.predict_f(XS, diagonal=True, squeeze=False)
        return var

    def covar(self, XS_1, XS_2, X=None, Y=None):
        raise NotImplementedError()

    def jittable_predict_f(self, XS, YS, nan_grid_X, nan_grid_Y, sort_idx, return_idx):
        """
        Due to the sorting required to into a spatio-temporal grid we require a separate prediction function that passes through the indexes required to sort.
        """
        raise NotImplementedError()
        X = self.raw_X
        Y = self.raw_Y
        X_stacked = np.vstack([X, XS, nan_grid_X])
        Y_stacked = np.vstack([Y, YS, nan_grid_Y])

        X_sorted = X_stacked[sort_idx]
        Y_sorted = Y_stacked[sort_idx]

        _, mu, var = filter_and_smooth(
            X_sorted.value,
            Y_sorted.value,
            self.kernel,
            self.likelihood,
            N = N
        )

        mu = mu.reshape([-1, 1])
        var = var.reshape([-1, 1])

        # unsort
        mu = mu[return_idx]
        var = var[return_idx]

        return mu, var

    def filter(self, data, prior, R = None, R_inv = None, full_state=False, return_lml = False, train_test_mask = None, train_index=None):
        lml, kf_res  = kalman_filter.filter_loop(
            data,
            prior,
            R = R,
            R_inv = R_inv,
            filter_type = self.filter_type,
            train_test_mask = train_test_mask,
            train_index=train_index
        ) 


        mu, var = kf_res['m'], kf_res['P']

        if return_lml:
            return lml, mu, var
        else:
            return mu, var

    def filter_and_smooth(self, data, prior, R = None, R_inv = None, full_state=False, return_lml = False, train_test_mask = None, train_index=None):
        lml, kf_res  = kalman_filter.filter_loop(
            data,
            prior,
            R = R,
            R_inv = R_inv,
            filter_type = self.filter_type,
            train_test_mask=train_test_mask,
            train_index=train_index
        ) 

        mu, var = rts_smoother.smoother_loop(
            data, 
            prior,
            kf_res,
            full_state=full_state,
            filter_type = self.filter_type
        )

        if return_lml:
            return lml, mu, var
        else:
            return mu, var

    def posterior_blocks(self, return_lml = False):
        """ Compute the posterior p(f_t | Y) for all t in time-latent-space format.  """
        R, R_inv = get_R_R_inv(self.likelihood)

        lml, mu, var = self.filter_and_smooth(
            self.data,
            self.prior,
            R = R,
            R_inv = R_inv,
            return_lml=True
        )
        chex.assert_rank([mu, var], [3, 3])

        # make var standard size of rank 4
        var = var[:, None, ...]

        # in time-latent-space format
        chex.assert_rank([mu, var], [3, 4])

        if return_lml:
            return lml, mu, var
        else:
            return mu, var

    def posterior(self, diagonal=True, full_state=False):
        R, R_inv = get_R_R_inv(self.likelihood)
        mu, var = self.filter_and_smooth(
            self.data,
            self.prior,
            R = R,
            R_inv = R_inv,
            full_state = full_state
        )

        # only fix shapes is not returning the full-state
        if full_state is False:
            # mu, var are in time - space format
            # Therefore we just need to stack them
            mu = np.reshape(mu, [-1, 1])

            # only keep diagonals
            if diagonal:
                var_diag = np.diagonal(var, axis1=1, axis2=2)
                var_diag = np.reshape(var_diag, [-1, 1])

                return mu, var_diag

        return mu, var



    def predict_blocks(self, XS, group_size, block_size, diagonal=False):
        chex.assert_equal(group_size, 1)

        NS = XS.shape[0]
        chex.assert_equal(XS.shape[1], self.data.D)

        X = onp.array(self.data.X)
        Y = onp.reshape(self.data.Y, [-1, self.output_dim])

        # Stack X first so that training data does not get removed when sorting data
        X_stacked = onp.vstack([X, XS])

        Y_nans = onp.NaN * onp.ones([NS, self.output_dim])
        Y_stacked = onp.vstack([Y, Y_nans])

        test_data = get_sequential_data_obj(
            X_stacked,
            Y_stacked,
            sort=True 
        )

        mu, var = self.filter_and_smooth(
            test_data,
            self.prior,
            R = self.get_likelihood_for_prediction(test_data)
        )

        # mu, var are in time - space format
        # Therefore we just need to stack them
        mu = np.reshape(mu, [-1, self.output_dim])
        var = np.reshape(var, [-1, self.output_dim, self.output_dim])

        # Unsort data and remove the training data
        mu = test_data.unsort(mu)[self.data.N:]
        var = test_data.unsort(var)[self.data.N:]

        return mu, var

    def predict_y(self, XS, diagonal=True, squeeze=True, force_full_state=False):
        if not diagonal :
            # TODO: only supports diagonal
            raise RuntimeError('Only diagonal predict_y is support')

        pred_mu, pred_var = self.predict_f(XS, squeeze=False, diagonal=True, force_full_state=force_full_state)
        chex.assert_rank([pred_mu, pred_var], [3, 4])

        pred_y_mu, pred_y_var = evoke('predict_y_diagonal', self, self.likelihood)(
            XS,  self.likelihood, pred_mu, pred_var
        )

        if squeeze:
            pred_y_mu = np.squeeze(pred_y_mu)
            pred_y_var = np.squeeze(pred_y_var)

        return pred_y_mu, pred_y_var

class T_SDE_GP(BASE_SDE_GP):
    """ Temporal SDE GP """

    def get_likelihood_for_prediction(self, data):
        # Currently the likelihood is only defined on the training points. 
        # But due to the implementation we need to provide likelihood values everywhere
        #  we need to be careful becasuse Jax will silently wraps 
        #  around in this setting if less data is passed through
        R = self.likelihood.variance

        out_dim = R.shape[-1]

        points_added = data.Nt-R.shape[0] 
        # Full R. This wont be used at locations without data so we just ignore
        R_tmp = np.tile(np.eye(out_dim), [points_added, 1, 1])
        R = np.vstack([R, R_tmp])
        R = R[data.unique_idx][data.sort_idx]
        R = np.reshape(R, [data.Nt, self.likelihood.block_size, self.likelihood.block_size])

        return R

    def get_train_test_index_and_mask(self, data):
        Nt = self.data.Nt
        points_added = data.Nt-Nt 
        unsorted_mask = np.hstack([np.ones(Nt), np.ones(points_added)*onp.NaN])
        unsorted_range = np.hstack([np.arange(Nt), np.ones(points_added).astype(np.integer)*-1]) # ones will be mapped to nan anyway, so doesnt matter what index get assigned 
        sorted_mask = unsorted_mask[data.unique_idx][data.sort_idx]
        train_index = unsorted_range[data.unique_idx][data.sort_idx]
        return train_index, sorted_mask

    def predict_f(self, XS: np.ndarray, diagonal=True, squeeze=False, filter_only: bool =False, force_full_state: bool = False):
        """
        Args:
            filter_only (bool): if True only run the Kalman filter, not smoother
            force_full_state (bool): If True we return the full state (ie without H)
        """

        if (filter_only == True) and (force_full_state == True):
            raise RuntimeWarning('filter_only alrady returns the full state')

        NS = XS.shape[0]
        chex.assert_equal(XS.shape[1], self.data.D)

        X = onp.array(self.data.X)
        Y = onp.reshape(self.data.Y_flat, [-1, self.output_dim])

        # Stack X first so that training data does not get removed when sorting data
        X_stacked = onp.vstack([X, XS])

        Y_nans = onp.NaN * onp.ones([NS, self.output_dim])
        Y_stacked = onp.vstack([Y, Y_nans])

        test_data = get_sequential_data_obj(
            X_stacked,
            Y_stacked,
            sort=True 
        )

        train_index, train_mask = self.get_train_test_index_and_mask(test_data)
        if filter_only:
            mu, var = self.filter(
                test_data,
                self.prior,
                R = self.get_likelihood_for_prediction(test_data),
                train_test_mask = train_mask,
                train_index = train_index
            )
            chex.assert_rank([mu, var], [3, 3])


        else:
            mu, var = self.filter_and_smooth(
                test_data,
                self.prior,
                R = self.get_likelihood_for_prediction(test_data),
                train_test_mask = train_mask,
                train_index = train_index,
                full_state = force_full_state
            )

            chex.assert_rank([mu, var], [3, 3])

        # fix mu and var shapes

        state_dim = mu.shape[1]

        # TODO: rename out_dim or self.output_dim
        if force_full_state or filter_only:
            out_dim = state_dim
        else:
            out_dim = self.output_dim


        # mu, var are in time - latent- space format but space is 1
        # Therefore we just need to stack them as no permutations are required
        mu = np.reshape(mu, [-1, out_dim])

        # Unsort data and remove the training data
        mu = test_data.unsort(mu)[self.data.N:]

        # fix var shape
        if diagonal:
            # var has rank 3
            var_diag = np.diagonal(var, axis1=1, axis2=2)
            var = np.reshape(var_diag, [-1, out_dim])

        var = test_data.unsort(var)[self.data.N:]

        if squeeze:
            mu = np.squeeze(mu)
            var = np.squeeze(var)
        else:
            # ensure rank 3 and 4
            # mu will be of rank 2
            # var will either be rank 2 or 3 depending on if diagonals are kept

            mu = mu[..., None]

            if diagonal:
                # add missing diagonal axid which is removed when extracting the diagonal
                var = var[..., None, None]
            else:
                var = var[:, None, ...]

            chex.assert_rank([mu, var], [3, 4])

        return mu, var

class ST_SDE_GP(BASE_SDE_GP):
    """ Spatio-Temporal SDE GP """

    def get_likelihood_for_prediction(self, data):
        # Currently the likelihood is only defined on the training points. 
        # But due to the implementation we need to provide likelihood values everywhere
        # Jax will silently wraps around in this setting if less data is passed through

        lik_R = self.likelihood.variance

        # Data is temporal data. We do not use the Kalman filter and smoother to predict in space,

        if isinstance(self.likelihood, ProductLikelihood) or issubclass(type(self.likelihood), ProductLikelihood):
            assert len(self.likelihood.likelihood_arr) == 1
            out_dim = self.likelihood.likelihood_arr[0].block_size

        else:
            out_dim = self.likelihood.block_size

        # We pad the train/test data with identity matrices
        # To pad in correct places we construct the train/test likelihood vars
        # in the same way the train-test data is constructed when predicting
        # then we can just use the same sorting indexes

        #points_added = data.Nt-lik_R.shape[0] 
        points_added = data.num_original_points-lik_R.shape[0] 
        # Full R. This wont be used at locations without data so we just ignore
        R_tmp = np.tile(np.eye(out_dim), [points_added, 1, 1])
        R_concat = np.vstack([lik_R, R_tmp])

        # sort using the same indexes as the train/test data
        pred_R = R_concat[data.unique_idx][data.sort_idx]
        chex.assert_shape(pred_R, [data.Nt, out_dim, out_dim])

        return pred_R

    def get_train_test_index_and_mask(self, data):
        Nt = self.data.Nt
        points_added = data.Nt-Nt 
        unsorted_mask = np.hstack([np.ones(Nt), np.ones(points_added)*onp.NaN])
        unsorted_range = np.hstack([np.arange(Nt), np.ones(points_added).astype(np.integer)*-1]) # ones will be mapped to nan anyway, so doesnt matter what index get assigned 
        sorted_mask = unsorted_mask[data.unique_idx][data.sort_idx]
        train_index = unsorted_range[data.unique_idx][data.sort_idx]
        return train_index, sorted_mask

    def predict_temporal(self, XS, filter_only=False):
        """
        Predicts in time locations in XS and at the spatial locations in the training data
        This is useful because Kalman filtering and smoothing algorithms are used to predict in time, and then spatial predictions is handled separately.

        When predicting we need to sort XS and the training data, but we need to be careful to not sort the spatial locations as these could be inducing points
            and so they need to be treated consistently.
        """
        YS_nans = onp.NaN * onp.ones([XS.shape[0], self.output_dim]) # dummy Y values

        # Convert XS to time-space format -- use TemporallyGroupedData for efficiency
        XS_st_data = TemporallyGroupedData(
            XS, 
            YS_nans,
            sort=True
        ) 

        # The KF is used to predict at new time points. Collect the order temporal points across
        #    trainig and testing data.
        # This will also be used to unsort the results
        # NOTE: self.data must go before XS_temporal_data otherwise data points can be overwritten by the sorting
        #    as only unique points are kept

        all_t = np.vstack([self.data.X_time[:, None], XS_st_data.X_time[:, None]])
        all_temporal_data = get_sequential_data_obj(
            all_t,
            np.ones_like(all_t), # Dummy data, we only care about X here
            sort=True
        )

        # we now need to add on the training spatial locations

        # use dummy values for space so that they do not get sorted
        training_data_with_dummy_space = get_sequential_data_obj(
            SpatialTemporalInput(
                self.data.X_time, 
                np.tile(np.arange(self.data.X_space.shape[0])[:, None], [1, self.data.X_space.shape[1]])
            ),
            self.data.Y_st,
            sort=False 
        )

        # Collect Training Data 
        X_with_dummy_space = onp.array(training_data_with_dummy_space.X)

        # self.data.Y is stored in time-space-latent format, reshape into data-latent
        Y_with_dummy_space = onp.reshape(training_data_with_dummy_space.Y_flat, [-1, self.output_dim])

        # create new data with the same spatial points as self.data but with the prediction time points 
        XS_with_dummy_space = add_temporal_points(XS_st_data, training_data_with_dummy_space)
        YS_with_dummy_space_nans = onp.NaN * onp.ones([XS_with_dummy_space.shape[0], self.output_dim]) # data-latent format

        # Stack X first so that training data does not get removed when sorting data
        X_all_with_dummy_space = onp.vstack([X_with_dummy_space, XS_with_dummy_space])
        Y_all_with_dummy_space = onp.vstack([Y_with_dummy_space, YS_with_dummy_space_nans])

        # ST data object across all (unique) training and testing temporal points but only 
        #   at the training spatial locations
        # This does not sort space because we dummy points for space that already ordered
        stacked_temporal_test_data = get_sequential_data_obj(
            X_all_with_dummy_space,
            Y_all_with_dummy_space,
            sort=True 
        )

        # we now need to replace the dummy spatial points with the actual ones
        # we now have a spatio-temporal grid where the spatial part is unchanged
        temporal_test_data = get_sequential_data_obj(
            SpatialTemporalInput(
                stacked_temporal_test_data.X_time, 
                self.data.X_space,
            ),
            np.transpose(np.reshape(stacked_temporal_test_data.Y_flat, [-1, self.data.Ns, self.data.P]), [0, 2, 1]),
            sort=False
        )

        R = self.get_likelihood_for_prediction(all_temporal_data)
        train_index, train_mask = self.get_train_test_index_and_mask(all_temporal_data)


        # Compute posterior at temporal_test_data
        if filter_only:
            mu_t, var_t = self.filter(
                temporal_test_data,
                self.prior,
                R = R,
                train_test_mask=train_mask,
                train_index=train_index
            )


            if not self.full_state_observed:
                # in time - latent - space - state
                # remove the extra state dims
                # assuming latent is 1
                _mu_t = np.copy(mu_t)

                # TODO: this is just a quick way to get the state size
                flat_mu_t = np.reshape(
                    mu_t,
                    [
                        temporal_test_data.Nt, 
                        self.prior.num_latents, 
                        self.data.Ns, 
                        -1
                    ]
                )
                state_size = flat_mu_t.shape[-1]

                mu_t = mu_t[:, ::state_size, ...]
                var_t = var_t[:, ::state_size, ...][:, :, ::state_size]
        else:
            mu_t, var_t = self.filter_and_smooth(
                temporal_test_data,
                self.prior,
                R = R,
                train_test_mask=train_mask,
                train_index=train_index
            )

        return XS_st_data, all_temporal_data, temporal_test_data, mu_t, var_t[:, None, ...]

    def _predict_f(self, XS: np.ndarray, diagonal=True, squeeze=False, sort_output = True, filter_only=False, force_full_state: bool = False):
        """
        We use the Kalman filter and smoother to predict and the temporal slices of XS,
        and then use the results to extrapolate to the new spatial locations.

        Whilst we could use the Kalman smoother to predict at all these locations, having them 
        separate requires less pre-processing of the data, and having them separate is required for
        CVI anyway.

        In:
            XS: Ns x D

        When diagonal is True we return
            mu;
            var:

        When diagonal is False we return the block diagonal across latents

        When filter_only is true we only return f from the filtering distributions

        Args:
            filter_only (bool): if True only run the Kalman filter, not smoother
            force_full_state (bool): If True we return the full state (ie without H)
        """
        chex.assert_equal(XS.shape[1], self.data.D)

        if not diagonal :
            if not force_full_state:
                if not self.full_state_observed:
                    raise RuntimeWarning('Currently we do not return the full predictive posterior covariance using an SDE GP')

        if force_full_state:
            # NOTE: we cannot support it as we need to predict in space, which would require a little more machinery implemented
            # NOTE: this is actually all implemented in the PIGP models, but there state-space predictions code is all separate, 
            #   perhaps we can combine at some point?
            raise RuntimeError('We do not support forcing full state, use full_state_observed when constructing the SDE_GP instead')

        # Convert XS to a spatio-temporal object
        NS = XS.shape[0]
        # in data-latent format
        YS_nans = onp.NaN * onp.ones([NS, self.output_dim]) # dummy Y values


        # TODO: this will be massive :( 
        # need to use temporally grouped data here...

        # Convert XS to time-space format
        XS_data = get_sequential_data_obj(
            XS, 
            YS_nans,
            sort=True
        )

        # The KF is used to predict at new time points. Collect the order temporal points across
        #    trainig and testing data.
        # This will also be used to unsort the results
        # NOTE: self.data must go before XS_data otherwise data points can be overwritten by the sorting
        #    as only unique points are kept
        all_t = np.vstack([self.data.X_time[:, None], XS_data.X_time[:, None]])
        all_temporal_data = get_sequential_data_obj(
            all_t,
            np.ones_like(all_t), # Dummy data, we only care about X here
            sort=True
        )

        # we construct dummy data using arange, as we do not require data to be spatially sorted, just 
        #  sorted in a time-space grid (this distinction is important when using spatial inducing points)
        dummy_training_data = get_sequential_data_obj(
            SpatialTemporalInput(
                self.data.X_time, 
                np.tile(np.arange(self.data.X_space.shape[0])[:, None], [1, self.data.X_space.shape[1]])
            ),
            self.data.Y_st,
            sort=False 
        )

        # Collect Training Data
        X = onp.array(dummy_training_data.X)

        # self.data.Y is stored in time-space-latent format, reshape into data-latent
        Y = onp.reshape(dummy_training_data.Y_flat, [-1, self.output_dim])

        # create new data with the same spatial points as self.data but with all time points across XS and X
        XS_temporal_new = add_temporal_points(XS_data, dummy_training_data)
        YS_temporal_new_nans = onp.NaN * onp.ones([XS_temporal_new.shape[0], self.output_dim]) # data-latent format

        # Stack X first so that training data does not get removed when sorting data
        X_stacked = onp.vstack([X, XS_temporal_new])
        Y_stacked = onp.vstack([Y, YS_temporal_new_nans])

        # ST data object across all (unique) training and testing temporal points but only 
        #   at the training spatial locations

        # this should not sort space!!
        # we should not be sorting space for test_data as this is also used for induicng points
        # . where ordering in space is not guarenteed
        # so we first order in time to get the unique points
        temporal_test_data = get_sequential_data_obj(
            X_stacked,
            Y_stacked,
            sort=True 
        )

        XS_temporal_new = add_temporal_points(XS_data, self.data)
        YS_temporal_new_nans = onp.NaN * onp.ones([XS_temporal_new.shape[0], self.output_dim]) # data-latent format

        # Stack X first so that training data does not get removed when sorting data
        Y_stacked = onp.vstack([self.data.Y, YS_temporal_new_nans])

        _X = X_stacked[temporal_test_data.unique_idx][temporal_test_data.sort_idx]
        _Y = Y_stacked[temporal_test_data.unique_idx][temporal_test_data.sort_idx]


        # we now have a spatio-temporal grid where the spatial part is unchanged
        temporal_test_data = get_sequential_data_obj(
            SpatialTemporalInput(
                temporal_test_data.X_time, 
                self.data.X_space,
            ),
            np.transpose(np.reshape(_Y, [-1, self.data.Ns, self.data.P]), [0, 2, 1]),
            sort=False
        )


        # Compute posterior at temporal_test_data
        if filter_only:
            mu_t, var_t = self.filter(
                temporal_test_data,
                self.prior,
                R = self.get_likelihood_for_prediction(all_temporal_data)
            )
            
            # in time - latent - space - state
            # remove the extra state dims
            # assuming latent is 1
            _mu_t = np.copy(mu_t)


            # TODO: this is just a quick way to get the state size
            flat_mu_t = np.reshape(
                mu_t,
                [
                    temporal_test_data.Nt, 
                    self.prior.num_latents, 
                    self.data.Ns, 
                    -1
                ]
            )
            state_size = flat_mu_t.shape[-1]
            mu_t = mu_t[:, ::state_size, ...]
            var_t = var_t[:, ::state_size, ...][:, :, ::state_size]


        else:
            mu_t, var_t = self.filter_and_smooth(
                temporal_test_data,
                self.prior,
                R = self.get_likelihood_for_prediction(all_temporal_data)
            )
            output_dim = self.output_dim

        # construct testing data at new spatial locations
        XS_spatial_new = add_temporal_points(all_temporal_data, XS_data)
        YS_spatial_new_nans = onp.NaN * onp.ones([XS_spatial_new.shape[0], self.output_dim])

        xs_spatial_data = get_sequential_data_obj(
            XS_spatial_new,
            YS_spatial_new_nans,
            sort=True 
        )

        # mu_t and var_t are in time-latent-space format
        # when predicting we only predict f, not the state as well
        if not self.full_state_observed:
            # remove the extra state dims
            mu_t = mu_t[:, :self.data.Ns, :]
            var_t = var_t[:, :self.data.Ns, :][:, :, :self.data.Ns]

        if not sort_output:
            # For certain models we do not want to actually unsort the prediction and this will be handled downstream

            # we need to return the data objects so that the unsorting can be performed
            return xs_spatial_data, all_temporal_data, XS_data, mu_t, var_t[:, None, ...]

        # Compute spatial conditions to get posterior at new spatial points
        mu, var = evoke('spatial_conditional', XS_data, temporal_test_data, self, self.prior)(
            xs_spatial_data, temporal_test_data, mu_t, var_t, self, False
        )


        # mu/var is in  time - (latent x space) format, need it in time-space-latent
        # Unsort data and remove the training data
        mu_time_unsorted = all_temporal_data.unsort(mu)[self.data.Nt:]
        var_time_unsorted = all_temporal_data.unsort(var)[self.data.Nt:]

        # convert to time-space-latent format
        # TODO: how are multiple latent functions handled here?
        if self.full_state_observed:
            out_dim = self.prior.spatial_output_dim*self.prior.temporal_output_dim

            mu_p = permute_vec_tps_to_tsp(mu_time_unsorted, num_latents=out_dim)
            # ensure rank 3
            var_p = permute_mat_tps_to_tsp(var_time_unsorted[:, 0, ...], num_latents=out_dim)

            mu_p = np.reshape(mu_p, [-1, out_dim, 1])
            var_p = batched_block_diagional(var_p, out_dim)
            var_p = np.reshape(var_p, [-1, 1, out_dim, out_dim])
        else:
            mu_p = mu_time_unsorted
            var_p = var_time_unsorted

            # time - space format
            mu_p = np.reshape(mu_p, [-1, 1, 1])
            var_p = np.reshape(
                np.diagonal(var_p, axis1=2, axis2=3),
                [-1, 1, 1, 1]
            )
        
        # unsort to original permutation in XS
        mu_p_unsorted = XS_data.unsort(mu_p)
        var_p_unsorted = XS_data.unsort(var_p)


        return mu_p_unsorted, var_p_unsorted

    def predict_f(self, XS: np.ndarray, diagonal=True, squeeze=False, sort_output = True, filter_only=False, force_full_state: bool = False):
        """
        We use the Kalman filter and smoother to predict and the temporal slices of XS,
        and then use the results to extrapolate to the new spatial locations.

        Whilst we could use the Kalman smoother to predict at all these locations, having them 
        separate requires less pre-processing of the data, and having them separate is required for
        CVI anyway.

        In:
            XS: Ns x D

        When diagonal is True we return
            mu;
            var:

        When diagonal is False we return the block diagonal across latents

        When filter_only is true we only return f from the filtering distributions

        Args:
            filter_only (bool): if True only run the Kalman filter, not smoother
            force_full_state (bool): If True we return the full state (ie without H)
        """
        chex.assert_equal(XS.shape[1], self.data.D)

        if not diagonal :
            if not force_full_state:
                if not self.full_state_observed:
                    raise RuntimeWarning('Currently we do not return the full predictive posterior covariance using an SDE GP')

        if force_full_state:
            # NOTE: we cannot support it as we need to predict in space, which would require a little more machinery implemented
            # NOTE: this is actually all implemented in the PIGP models, but there state-space predictions code is all separate, 
            #   perhaps we can combine at some point?
            raise RuntimeError('We do not support forcing full state, use full_state_observed when constructing the SDE_GP instead')

        xs_spatial_data, all_temporal_data, stacked_temporal_test_data, mu_t, var_t = self.predict_temporal(XS, filter_only=filter_only)
        var_t = var_t[:, 0, ...]

        # mu_t and var_t are in time-latent-space format
        # when predicting we only predict f, not the state as well
        if not self.full_state_observed:
            # remove the extra state dims
            mu_t = mu_t[:, :self.data.Ns, :]
            var_t = var_t[:, :self.data.Ns, :][:, :, :self.data.Ns]

        if not sort_output:
            # For certain models we do not want to actually unsort the prediction and this will be handled downstream

            # we need to return the data objects so that the unsorting can be performed
            return xs_spatial_data, all_temporal_data, XS_data, mu_t, var_t[:, None, ...]

        mu_t = all_temporal_data.unsort(mu_t)[self.data.Nt:]
        var_t = all_temporal_data.unsort(var_t)[self.data.Nt:]

        # data_x.X_time is not used, so  we can just pass the stacked temporal dataset

        # Compute spatial conditions to get posterior at new spatial points
        pred_mu, pred_var = evoke('spatial_conditional', xs_spatial_data, all_temporal_data, self, self.prior)(
            xs_spatial_data, stacked_temporal_test_data, mu_t, var_t, self, False
        )


        # convert to time-space-latent format
        # TODO: how are multiple latent functions handled here?
        if self.full_state_observed:
            out_dim = self.prior.spatial_output_dim*self.prior.temporal_output_dim

            mu_p = permute_vec_tps_to_tsp(pred_mu, num_latents=out_dim)
            # ensure rank 3
            var_p = permute_mat_tps_to_tsp(pred_var[:, 0, ...], num_latents=out_dim)

            mu_p = np.reshape(mu_p, [-1, out_dim, 1])
            var_p = batched_block_diagional(var_p, out_dim)
            var_p = np.reshape(var_p, [-1, 1, out_dim, out_dim])
        else:
            mu_p = pred_mu
            var_p = pred_var

            # time - space format
            mu_p = np.reshape(mu_p, [-1, 1, 1])
            var_p = np.reshape(
                np.diagonal(var_p, axis1=2, axis2=3),
                [-1, 1, 1, 1]
            )
        
        if True:
            # unsort to original permutation in XS
            mu_p_unsorted = xs_spatial_data.unsort(mu_p)
            var_p_unsorted = xs_spatial_data.unsort(var_p)


        return mu_p_unsorted, var_p_unsorted
