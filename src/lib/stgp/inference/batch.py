"""Batch inference class."""
from . import Inference
from ..dispatch import evoke


class Batch(Inference):
    """Batch inference class."""

    def neg_log_marginal_likelihood(self, data, gp, likelihood, prior):
        lml = evoke('log_marginal_likelihood', data, gp, likelihood, prior)(
            data, gp, likelihood, prior
        )

        return - lml

    def predict_f(self, XS, data, gp, likelihood, prior, diagonal: bool):

        pred_mu, pred_var = evoke('predict', data, gp, likelihood, prior)(
            XS, data, gp, likelihood, prior, diagonal
        )

        return pred_mu, pred_var

    def predict_f_blocks(self, XS, data, gp, likelihood, prior, group_size, block_size):

        pred_mu, pred_var = evoke('predict_blocks', data, gp, likelihood, prior)(
            XS, data, gp, likelihood, prior, block_size
        )

        return pred_mu, pred_var

    def predict_y(self, XS, data, gp, likelihood, prior, diagonal: bool):

        pred_mu, pred_var = self.predict_f(XS, data, gp, likelihood, prior, diagonal)

        pred_y_mu, pred_y_var = evoke('predict_y', gp, likelihood, prior)(
            XS, gp, likelihood, pred_mu, pred_var, diagonal
        )

        return pred_y_mu, pred_y_var

    def predictive_covar(self, XS_1, XS_2, data, gp, likelihood, prior):

        pred_var = evoke('predict_covar', data, gp, likelihood, prior)(
            XS_1, XS_2, data, gp, likelihood, prior
        )

        return pred_var
