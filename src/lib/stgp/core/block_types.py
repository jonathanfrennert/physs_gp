"""
Enum definitions for block types.

When handling both multi-task and spatio-temporal data various block sizes are required.

To keep track we use these enums.
"""

from enum import Enum
from ..dispatch import _ensure_str

class Block(Enum):
    DIAGONAL = 1
    FULL = 2
    SPATIAL = 3
    LATENT = 4
    LATENT_SPATIAL = 5
    BLOCK = 6 # GENERIC BLOCKS, will be implied by data or input shapes
    OUTPUT = 7
    

def get_block_dim(block_type: Block, data = None, likelihood = None, approximate_posterior = None):
    """ 
    Helper function to get actual block sizes

    When block_type is Diagonal the block size is always 1.

    When block_type is Block it can be 2 reasons:
        Block Diagonal Likelihood
        Full Gaussian Approximate Posterior
    """
    if block_type == Block.DIAGONAL:
        return 1

    if block_type == Block.BLOCK:
        if likelihood is not None:
            if _ensure_str(likelihood) in ['BlockDiagonalLikelihood', 'BlockDiagonalGaussian', 'PrecisionBlockDiagonalGaussian']:
                return likelihood.block_size

        if approximate_posterior is not None:
            return approximate_posterior.num_latents

    if block_type == Block.LATENT:
        if approximate_posterior is not None:
            return approximate_posterior.num_latents

    raise RuntimeError()

def get_block_type(shape: int) -> Block:
    if (shape == 1):
        return Block.DIAGONAL
    
    return Block.BLOCK

def compare_block_types(b1: Block, b2: Block) -> Block:
    """
    Returns the `greater' block size of b1 and b2
    """

    order = [Block.DIAGONAL, Block.LATENT, Block.BLOCK,  Block.FULL]

    return order[
        max(
            order.index(b1),
            order.index(b2)
        )
    ]

