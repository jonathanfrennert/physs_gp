from . trainer import Trainer, ScipyTrainer, GradDescentTrainer, SwitchTrainer

from .natgrad_trainer import NatGradTrainer

__all__ = [
    'Trainer', 
    'ScipyTrainer',
    'NatGradTrainer',
    'GradDescentTrainer',
    'SwitchTrainer'
]
