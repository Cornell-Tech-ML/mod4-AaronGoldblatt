from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # Implemented for Task 4.3.
    # Calculate the new height and width after pooling using integer division
    new_height = height // kh
    new_width = width // kw
    # Reshape to split both height and width dimensions into their kernel components
    # From: (batch, channel, height, width)
    # To:   (batch, channel, new_height, kh, new_width, kw)
    input = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    # Rearrange dimensions to get pooling windows together
    # From: (batch, channel, new_height, kh, new_width, kw)
    # To:   (batch, channel, new_height, new_width, kh * kw)
    input = input.permute(0, 1, 2, 4, 3, 5).contiguous()
    input = input.view(batch, channel, new_height, new_width, kh * kw)
    return input, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling 2D over a tensor using a given kernel size

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width that has been averaged over the pooling kernel
        
    """
    # Get the shape of the input tensor
    batch, channel, _, _ = input.shape
    # Reshape the input tensor using the tile function
    tiled, new_height, new_width = tile(input, kernel)
    # Average over the pooling window and ensure correct output shape
    return tiled.mean(dim=-1).view(batch, channel, new_height, new_width)


# TODO: Implement for Task 4.4.
