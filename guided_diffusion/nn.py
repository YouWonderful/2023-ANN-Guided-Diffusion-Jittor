"""
Various utilities for neural networks.
"""

import math

import jittor as jt
from jittor import nn
from jittor import init

# Jittor doesn't have AvgPool1d
class AvgPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
    
    def execute(self, x):
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html
        # Calculate output size
        input_size = x.size(2)
        output_size = (input_size + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Pad input tensor
        x = nn.pad(x, (self.padding, self.padding))
        
        # Unfold input tensor and perform average pooling
        unfolded_input = nn.unfold(x.unsqueeze(0), self.kernel_size, self.stride)
        unfolded_input = unfolded_input.view(1, x.size(1), self.kernel_size, -1)
        
        # Calculate average
        if self.count_include_pad:
            divisor = self.kernel_size
        else:
            divisor = unfolded_input.sum(2)
            divisor[divisor == 0] = 1  # Avoid division by zero
            divisor = divisor.unsqueeze(2)
        
        output = unfolded_input.sum(dim=2) / divisor
        
        # Reshape the result tensor
        output = output.view(1, x.size(1), output_size, -1)
        
        return output

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def execute(self, x):
        return x * jt.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def execute(self, x):
        return super().execute(x.float()).unary(op=x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.mul_(rate).mul_(1 - rate).add_(src) # alpha=1 - rate


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dims=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = jt.exp(
        -math.log(max_period) * jt.arange(start=0, end=half, dtype=jt.float32) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = jt.concat([jt.cos(args), jt.sin(args)], dim=-1)
    if dim % 2:
        embedding = jt.concat([embedding, jt.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    # if flag:
    #     args = tuple(inputs) + tuple(params)
    #     return CheckpointFunction.apply(func, len(inputs), *args)
    # else:
    return func(*inputs)


class CheckpointFunction(jt.Function):
    def execute(self, run_function, length, *args):
        self.run_function = run_function
        self.input_tensors = list(args[:length])
        self.input_params = list(args[length:])
        with jt.no_grad():
            output_tensors = self.run_function(*self.input_tensors)
        return output_tensors

    def grad(self, *output_grads):
        self.input_tensors = [x for x in self.input_tensors]
        with jt.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in self.input_tensors]
            output_tensors = self.run_function(*shallow_copies)
        # Jittor is auto-grad
        input_grads = jt.grad(
            output_tensors,
            self.input_tensors + self.input_params,
            # output_grads,
            # allow_unused=True,
        )
        del self.input_tensors
        del self.input_params
        del output_tensors
        return (None, None) + input_grads