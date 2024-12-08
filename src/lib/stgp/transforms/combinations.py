"""General combination and DGP specific transforms."""
from . import Transform


class Zip(Transform):
    """Apply transform to each input in order specified."""

    pass


class FullyConnected(Transform):
    """Pass each input through each transform."""

    pass
