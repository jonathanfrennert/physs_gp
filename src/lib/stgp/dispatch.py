import types
import inspect

from enum import Enum

class DispatchNotFound(Exception):
    def __init__(self, s):
        self.s = s

_REGISTERED = {}

def _is_arg_more_specific(arg1, arg2):
    # If the argument of arg1 or arg2 is a string they must be the same
    #   as we cannot do inhertentence checking on strings
    if type(arg1) == str or type(arg2) == str:
        if _ensure_str(arg1) != _ensure_str(arg2):
            # Cannot compare dispatched types with strings therefore assume that it is not more specific
            return False

    # If arg1 and arg2 are the same class then it is more specific
    elif arg1 == arg2:
        return True

    # If arg1 is not a subclass of arg2 this means that it must be more general hence we can return False
    elif not issubclass(arg1, arg2):
        return False

    return True

def is_more_specific(key_1, key_2):
    """
    Returns true if key_1 is more specific than key_2
    key_1 is only more specific if every arg and kwargs is a subclass or the same class as the 
        correpondigs (k)args in key_2
    """

    specific_flag = True


    for i in range(len(key_1.args)):
        if _is_arg_more_specific(key_1.args[i], key_2.args[i]):
            continue
        else:
            return False

    for k, v in key_1.kwargs.items():
        if _is_arg_more_specific(key_1.kwargs[k], key_2.kwargs[k]):
            continue
        else:
            return False


    return True

def _ensure_str(k):
    if type(k) is not str:
        #the passed k is either a class or an class instance / object

        if not isinstance(k, type):
            #the passed k in an object
            k = type(k)

        return k.__name__
    return k

def _try_match(x, y):

    # special cases
    if isinstance(x, bool) and isinstance(y, bool) :
        return x == y

    if isinstance(x, Enum) and isinstance(y, Enum) :
        return x == y

    # check if y is a child class of x
    if inspect.isclass(x):
        _x = x
    else:
        _x = type(x)

    if inspect.isclass(y):
        _y = y  
    else:
        _y = type(y)

    # avoid catch alls
    if _x != object and _y != object:
        if _x != _y:
            # only check for inherentance
            # as string comparision will catch same types

            if issubclass(_y, _x):
                return True

    if _ensure_str(x) != _ensure_str(y):
        return False

    return True


class _REGISTERED_KEY:
    def __init__(self, obj, args, kwargs):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs

class _DISPATCHER:
    @staticmethod
    def match(key:_REGISTERED_KEY, *args, **kwargs):
        if len(key.args) != len(args):
            return False

        if len(key.kwargs.keys()) != len(kwargs.keys()):
            return False

        for x, y in zip(key.args, args):
            if not _try_match(x, y):
                return False

        for k, i in key.kwargs.items():
            if _ensure_str(k) not in kwargs.keys():
                return False

            if not _try_match(kwargs[k], key.kwargs[k]):
                return False

        return True

def dispatch(*args, **kwargs):
    def decorator(obj):
        if isinstance(obj, types.FunctionType):
            # dispatch a function

            # add fn name to the key
            k = _REGISTERED_KEY(obj, [obj.__name__, *args], kwargs)

        elif inspect.isclass(obj):
            # dispatch a class
            # Although the class is passed it will not be used to construct the key
            k = _REGISTERED_KEY(obj, args, kwargs)

        else:
            raise DispatchNotFound(f'Cannot Dispatch {obj}')

        _REGISTERED[k] = obj

        return obj

    return decorator

def evoke(*args, debug=False, **kwargs):
    matched_item = None
    matched_key = None

    if debug:
        print('============= called from ====')
        print(f"{inspect.stack()[1].filename} -- {inspect.stack()[1].lineno}")
        print('=================')

    for k, item in _REGISTERED.items():


        if _DISPATCHER.match(k, *args, **kwargs):
            if debug:
                print('=============')
                print('new match')
                print(item)
                print(k)
                print(k.args)
            if matched_item is None or is_more_specific(k, matched_key):
                if debug:
                    print('most specific')

                matched_item = item
                matched_key = k


    if matched_item != None:
        if debug:
            print('=============')
            print('matched item')
            print(matched_item)
            print(f"{matched_item.__globals__['__file__']} -- {inspect.getsourcelines(matched_item)[1]}")
            print(inspect.getfullargspec(matched_item))
            print(matched_key.args)
            print('=============')
        return matched_item

    raise DispatchNotFound(f'Cannot evoke {args}, {kwargs}')
