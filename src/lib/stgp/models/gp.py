from ..dispatch import dispatch, evoke
from ..inference import Batch
from ..core import GPPrior

def GP(*args, inference=Batch(), **kwargs):
    # Y is passed either explictly through kwargs to implitly through args
    if 'Y' in kwargs.keys() or 'data' in kwargs.keys() or len(args) > 1:
        return evoke('Model', inference)(*args, **kwargs)
    else:
        return GPPrior(*args, **kwargs)

