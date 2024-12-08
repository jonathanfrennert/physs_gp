""" Collection of standard Trainers """

import objax
import jax

from .trainer import Trainer, ScipyTrainer, GradDescentTrainer, SwitchTrainer
from .natgrad_trainer import NatGradTrainer



class LBFGS(ScipyTrainer):
    """ Constructs a ScipyTrainer with L-BFGS-B argument """
    def __init__(self, *args, **kwargs):
        super(LBFGS, self).__init__(*args, optimizer='L-BFGS-B', **kwargs)

class ADAM(GradDescentTrainer):
    """ Constructs a GradDescentTrainer using Adam argument """
    def __init__(self, *args, **kwargs):
        super(ADAM, self).__init__(*args, optimizer=objax.optimizer.Adam, **kwargs)

class VB_NG_LBFGS(Trainer):
    """ 
    A variational bayes algorithm that uses Natural gradients to update the approximate posterior and LBFGS for the rest of the parameters.
    """
    def __init__(self, m, enforce_psd_type = None, ng_schedule=None, ng_nan_max_attempt=None):
        self.ng_trainer = NatGradTrainer(m, enforce_psd_type = enforce_psd_type, schedule=ng_schedule, nan_max_attempt=ng_nan_max_attempt, return_objective=False)

        # do not use adam to train the approximate posterior
        m.approximate_posterior.fix()
        self.lbfgs_trainer = LBFGS(m)

        # construct switch trainer to iteratively update the above
        self.switch_trainer = SwitchTrainer(
            [self.lbfgs_trainer, self.ng_trainer],
            callback_idx=0
        )

    def train(
        self,
        learning_rates: list,
        epochs: list,
        callback = None,
        raise_error = True,
        ng_momentum=False,
        ng_momentum_rate = None
    ):
        """
        Args:
           learning_rates: list: [adam_lr, ng_lr] 
           epochs: list: [num_epochs, [adam_iters, ng_iters]] 
        """
        # first update the natural gradients, this is so the first gradients from adam are starting from a 'good point'
        self.ng_trainer.train(learning_rates[1], epochs[1][1], raise_error=raise_error)
        return self.switch_trainer.train(learning_rates, epochs, callback, raise_error=raise_error, trainer_kwargs=[{}, {'momentum': ng_momentum, 'momentum_rate': ng_momentum_rate}])



class VB_NG_ADAM(Trainer):
    """ 
    A variational bayes algorithm that uses Natural gradients to update the approximate posterior and Adam for the rest of the parameters.
    """
    def __init__(self, m, enforce_psd_type = None, ng_schedule=None, ng_nan_max_attempt=None):
        self.ng_trainer = NatGradTrainer(m, enforce_psd_type = enforce_psd_type, schedule=ng_schedule, nan_max_attempt=ng_nan_max_attempt, return_objective=False)

        # do not use adam to train the approximate posterior
        m.approximate_posterior.fix()
        self.adam_trainer = GradDescentTrainer(m, objax.optimizer.Adam)

        # construct switch trainer to iteratively update the above
        self.switch_trainer = SwitchTrainer(
            [self.adam_trainer, self.ng_trainer],
            callback_idx=0
        )

    def train(
        self,
        learning_rates: list,
        epochs: list,
        callback = None,
        raise_error = True,
        ng_momentum=False,
        ng_momentum_rate = None
    ):
        """
        Args:
           learning_rates: list: [adam_lr, ng_lr] 
           epochs: list: [num_epochs, [adam_iters, ng_iters]] 
        """
        # first update the natural gradients, this is so the first gradients from adam are starting from a 'good point'
        self.ng_trainer.train(learning_rates[1], epochs[1][1], raise_error=raise_error)
        return self.switch_trainer.train(learning_rates, epochs, callback, raise_error=raise_error, trainer_kwargs=[{}, {'momentum': ng_momentum, 'momentum_rate': ng_momentum_rate}])

class LikNoiseSplitTrainer(Trainer):
    """ A trainer that holds the likelihood noise for a percentage of the training epochs """
    def __init__(self, m, trainer_wrapper, hold_noise_percent):
        m.likelihood.fix()
        self.trainer_with_lik_held = trainer_wrapper()
        m.likelihood.release()
        self.trainer_with_lik_released = trainer_wrapper()
        self.hold_percent = hold_noise_percent
        
    def train(
        self,
        learning_rates: list,
        epochs: list,
        callback = None,
        verbose = False,
        raise_error=False
    ):
        """
        Args:
            epochs: list: [num_epochs, *] 
        """

        # TODO: ensure valid values here
        if type(epochs) is not list:
            single_epoch = True
            max_iters = epochs
        else:
            single_epoch = False
            max_iters = epochs[0]
    

        iters_with_lik_held = int(self.hold_percent * max_iters)
        iters_with_lik_released = max_iters -  iters_with_lik_held


        if verbose:
            # wrap with empty prints to add new lines to helping viewing when using callbacks
            print('')
            print(f'training with likelihood held for {iters_with_lik_held}/{max_iters}')
            print('')

        if single_epoch:
            lc_1, _ = self.trainer_with_lik_held.train(learning_rates, iters_with_lik_held, callback, raise_error=raise_error)

        else:
            lc_1, _ = self.trainer_with_lik_held.train(learning_rates, [iters_with_lik_held, epochs[1]], callback, raise_error=raise_error)

        if verbose:
            print('')
            print(f'training with likelihood released for {iters_with_lik_released}/{max_iters}')
            print('')

        if single_epoch:
            lc_2, _ = self.trainer_with_lik_released.train(learning_rates, iters_with_lik_released, callback, raise_error=raise_error)
        else:
            lc_2, _ = self.trainer_with_lik_released.train(learning_rates, [iters_with_lik_released, epochs[1]], callback, raise_error=raise_error)

        return [lc_1, lc_2], None
