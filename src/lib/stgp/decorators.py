"""decorators used in gpajx."""
import inspect
from .errors import StrictModeError
from . import settings

def strict_mode_check(func):
    """Enforce strict model when settings.in_strict_mode is True.

    When in strict mode, if funchas optional params that are not passed then
        this decorator will raise a RunTimeError
    """
    # Retrieve the original argspec
    args = inspect.getfullargspec(func).args

    #ignore 'self' when func is part of a class
    if inspect.ismethod(func):
        args = args[1:]

    num_args = len(args)

    def inner(*args, **kwargs):

        if settings.in_strict_mode:
            if len(args) + len(kwargs.keys()) < num_args:
                # there must be a default argument being used
                raise StrictModeError()

        return func(*args, **kwargs)

    return inner


def cite(argument):
    """Add citation if the wrapped function is called."""

    def decorator(function):
        def inner(*args, **kwargs):
            inner.count += 1

            if inner.count == 1:
                settings.add_citation(argument)

            return function(*args, **kwargs)

        inner.count = 0
        return inner

    return decorator

def set_defaults_from_self(func):
    """Replace any unpassed variables with their value in self."""
    argspec = inspect.getfullargspec(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        #get arguments that have not been passed a value
        unpassed_positional_args = argspec.args[len(args):]

        #args[0] is the object (`self`), so get the unpassed value from self
        new_args = []
        for a in unpassed_positional_args:
            if a not in kwargs:
                if hasattr(args[0], a):
                    new_args.append((a, getattr(args[0], a)))
                else:
                    #use the default
                    pass

        kwargs.update(new_args)

        #call function with defaults from self
        return func(*args, **kwargs)

    return wrapper

def ensure_data(func):
    pass
