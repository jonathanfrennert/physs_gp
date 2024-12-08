import objax
from typing import Callable, Optional

from .computation.parameter_transforms import inv_positive_transform, positive_transform


class Parameter(objax.Module):
    """
    A wrapper around objax Trainvar to support naming a Parameter (for pretty printing of models)
        and to make constraints easier to use.
    """
    _NAME_DICT = {}

    def __init__(
        self, 
        val, 
        constraint:Optional[str]=None, 
        constraint_fn:Optional[Callable] = None, 
        inv_constraint_fn:Optional[Callable] = None, 
        name: Optional[str] = None,
        train: bool = True
    ):
        self.constraint = constraint
        self.constraint_fn = constraint_fn
        self.inv_constraint_fn = inv_constraint_fn

        if train:
            self.raw_var = objax.TrainVar(self.inv_transform(val))
        else:
            self.raw_var = objax.StateVar(self.inv_transform(val))

        self._train_state = train

        self.set_name(name)


    def set_name(self, name):
        if name is not None:
            if name in Parameter._NAME_DICT:
                Parameter._NAME_DICT[name] += 1
                self.name = f'{name} - {Parameter._NAME_DICT[name]}'
            else:
                Parameter._NAME_DICT[name] = 1
                self.name = name
        else:
            self.name = None

    def assign(self, val):
        self.raw_var.assign(self.inv_transform(val))

    @property
    def is_trainable(self):
        return self._train_state

    def fix(self):
        self._train_state = False

    def release(self):
        self._train_state = True
    
    @property
    def value(self):
        return self.transform(self.raw_var.value)
    
    def transform(self, var):
        if self.constraint_fn is not None:
            return self.constraint_fn(var)

        if self.constraint == None:
            return var
        elif self.constraint == 'positive':
            return positive_transform(var)

        raise RuntimeError(f'Constraint {self.constraint} is not supported!')

    def inv_transform(self, val):
        """
        Inverse transform val to the parameter space.
            If constraint_fn is passed this is a user constraint and so use the user defined inv_constraint_fn
            otherwise find the correct inverse function.
        """

        if self.constraint_fn is not None:
            return self.inv_constraint_fn(val)

        if self.constraint == None:
            return val
        elif self.constraint == 'positive':
            return inv_positive_transform(val)

        raise RuntimeError(f'Constraint {self.constraint} is not supported!')

