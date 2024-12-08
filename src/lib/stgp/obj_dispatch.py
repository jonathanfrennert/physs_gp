import numpy as np
from pydoc import locate
from .inference import Inference

_REGISTERED = {}

def _ensure_str(k):
    if type(k) is not str:
        #the passed k is either a class or an class instance / object

        if not isinstance(k, type):
            #the passed k in an object
            k = type(k)

        return k.__name__
    return k


class _REGISTERED_KEY:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

class _DISPATCHER:
    @staticmethod
    def match(key:_REGISTERED_KEY, *args, **kwargs):
        for x, y in zip(key.args, args):
            if _ensure_str(x) != _ensure_str(y):
                return False

        for k, i in key.kwargs.items():
            if _ensure_str(k) not in kwargs.keys():
                return False

            if _ensure_str(kwargs[k]) != _ensure_str(key.kwargs[k]):
                return False

        return True


def dispatch(*args, **kwargs):
    def decorator(obj):
        k = _REGISTERED_KEY(args, kwargs)
        _REGISTERED[k] = obj

        return obj

    return decorator

def evoke(*args, **kwargs):
    for k, item in _REGISTERED.items():
        if _DISPATCHER.match(k, *args, **kwargs):
            return item

    raise RuntimeError()
