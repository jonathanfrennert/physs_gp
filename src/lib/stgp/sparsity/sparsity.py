"""
Sparsity accepts a numpy array or a parameter object.
"""
import objax
import jax.numpy as np
from batchjax import batch_or_loop
from ..utils.utils import get_batch_type

from ..data import Input, SpatialTemporalInput
from ..parameter import Parameter

class Sparsity(Input):
    @property
    def X(self):
        return self.Z



class FreeSparsity(Sparsity):
    pass

class StructuredSparsity(Sparsity):
    pass

class NoSparsity(Sparsity):
    def __init__(self, Z: np.ndarray = None, Z_ref: Parameter = None):

        if Z is not None:
            self.raw_Z = Parameter(np.array(Z), constraint=None, train=False, name='Z')
        else:
            self.raw_Z = Z_ref

    @property
    def Z(self):
        return self.raw_Z.value

    def fix(self):
        """ NoSparsity is used for data, do not train data. """
        pass

    def release(self):
        pass

class FullSparsity(FreeSparsity):
    def __init__(self, Z: np.ndarray = None, Z_ref: Parameter = None):

        if Z is not None:
            self.raw_Z = Parameter(np.array(Z), constraint=None, name='Z')
        else:
            self.raw_Z = Z_ref

    @property
    def Z(self):
        return self.raw_Z.value

    def fix(self):
        """ Hold all parameters.  """
        self.raw_Z.fix()

    def release(self):
        """ Un-hold all parameters.  """
        self.raw_Z.release()


class SpatialSparsity(StructuredSparsity):
    def __init__(self, X_time = None, Z_space = None, Z_ref:SpatialTemporalInput = None, train=True):

        self.train_flag = train

        if Z_ref:
            self.raw_Z = Z_ref
        else:
            self.raw_Z = SpatialTemporalInput(
                X_time = X_time,
                X_space = Z_space,
                train=train
            )

    @property
    def Z(self):
        return self.raw_Z.value

    def fix(self):
        """ Hold all parameters.  """
        if self.train_flag:
            self.raw_Z.fix()

    def release(self):
        """ Un-hold all parameters.  """
        # only release if it was set a trainable parameter from the start
        if self.train_flag:
            self.raw_Z.release()

class _StackedSparsity(Sparsity):
    def __init__(self, sparsity_arr):
        self.sparsity_arr = objax.ModuleList(sparsity_arr)

    @property
    def Z(self):
        return batch_or_loop(
            lambda Z: Z.Z,
            [self.sparsity_arr],
            [0],
            dim=len(self.sparsity_arr),
            out_dim=1,
            batch_type = get_batch_type(self.sparsity_arr)
        )

class StackedSparsity(Sparsity):
    def __init__(self, sparsity_arr):
        self.sparsity_arr = objax.ModuleList(sparsity_arr)

    @property
    def Z(self):
        Z_arr =  batch_or_loop(
            lambda Z: Z.Z,
            [self.sparsity_arr],
            [0],
            dim=len(self.sparsity_arr),
            out_dim=1,
            batch_type = get_batch_type(self.sparsity_arr)
        )
        Z_arr = np.array(Z_arr)

        return np.reshape(Z_arr, [-1, Z_arr.shape[-1]])


class StackedNoSparsity(StackedSparsity, NoSparsity):
    pass
