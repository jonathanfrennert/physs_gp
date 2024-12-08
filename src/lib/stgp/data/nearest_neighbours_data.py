from . import Data

class NearestNeighboursData(Data):
    pass


class PrecomputedGroupedNearestNeighboursData(NearestNeighboursData):
    """
    Each data point is assoicated with a set of nearest points (groups) that the latent
        GP is defined over.

    This class wraps an existing data object
    """
    def __init__(self, data, neighbour_arr = None):
        """
        Args:
            neighbour_arr: array of arrays of indices. This is organised in the same format as data (ie data or time-space)
        """
        if neighbour_arr is None:
            raise RuntimeError('Neighbours must be passed!')

        self.base_data = data
        self.neighbour_arr = neighbour_arr
        self.N = self.base_data.N
        self.minibatch = self.base_data.minibatch


