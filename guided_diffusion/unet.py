from abc import abstractmethod

import math

import numpy as np
import jittor as jt
import jittor.nn as nn

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    SiLU
)

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            jt.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def execute(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = jt.concat([x.mean(dim=-1, keepdims=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where execute() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def execute(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def execute(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def execute(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = nn.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = nn.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def execute(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class EmbedBlock(nn.Module):
    """
    abstract class
    """
    @abstractmethod
    def execute(self, x, temb, cemb):
        """
        abstract method
        """
        
        
class EmbedSequential(nn.Sequential, EmbedBlock):
    def execute(self, x:jt.Var, temb:jt.Var, cemb:jt.Var) -> jt.Var:
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, temb, cemb)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def execute(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # return checkpoint(
        #     self._forward, (x, emb), self.parameters(), self.use_checkpoint
        # )
        return self._forward(x, emb)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = x
            for ly in in_rest:
                h = ly(h)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).unary(op=h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = jt.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            for ly in out_rest:
                h = ly(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class ClassifierFreeResBlock(EmbedBlock):
    def __init__(self, in_ch:jt.Var, out_ch:jt.Var, tdim:int, cdim:int, droprate:float):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tdim = tdim
        self.cdim = cdim
        self.droprate = droprate

        self.block_1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1),
        )

        self.temb_proj = nn.Sequential(
            SiLU(),
            nn.Linear(tdim, out_ch),
        )
        self.cemb_proj = nn.Sequential(
            SiLU(),
            nn.Linear(cdim, out_ch),
        )
        
        self.block_2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            SiLU(),
            nn.Dropout(p = self.droprate),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
            
        )
        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size = 1, stride = 1, padding = 0)
        else:
            self.residual = nn.Identity()
    def execute(self, x:jt.Var, temb:jt.Var, cemb:jt.Var) -> jt.Var:
        latent = self.block_1(x)
        latent += self.temb_proj(temb)[:, :, None, None]
        latent += self.cemb_proj(cemb)[:, :, None, None]
        latent = self.block_2(latent)

        latent += self.residual(x)
        return latent


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def execute(self, x):
        # return checkpoint(self._forward, (x,), self.parameters(), True)
        return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += jt.float64([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def execute(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jt.linalg.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = nn.softmax(weight.float(), dim=-1).unary(weight.dtype)
        a = jt.linalg.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def execute(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jt.linalg.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = nn.softmax(weight.float(), dim=-1).unary(weight.dtype)
        a = jt.linalg.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_classifier_free_diffusion=False,
        class_num=10
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = jt.float16 if use_fp16 else jt.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        self.use_classifier_free_diffusion = use_classifier_free_diffusion
        if self.use_classifier_free_diffusion:
            self.cemb_layer = nn.Sequential(
                nn.Linear(class_num, time_embed_dim),
                SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        if self.use_classifier_free_diffusion:
            self.input_blocks = nn.ModuleList(
                [EmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
            )
        else:
            self.input_blocks = nn.ModuleList(
                [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
            )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                if not self.use_classifier_free_diffusion:
                    layers = [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=int(mult * model_channels),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                else:
                    layers = [
                        ClassifierFreeResBlock(ch, int(mult * model_channels), time_embed_dim, time_embed_dim, self.dropout)
                    ]
                ch = int(mult * model_channels)
                if not self.use_classifier_free_diffusion:
                    if ds in attention_resolutions:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                            )
                        )
                else:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if not self.use_classifier_free_diffusion:
                    self.input_blocks.append(TimestepEmbedSequential(*layers))
                else:
                    self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                if self.use_classifier_free_diffusion:
                    self.input_blocks.append(EmbedSequential(ClassifierFreeDownsample(ch, conv_resample, dims=dims, out_channels=ch)))
                    input_block_chans.append(ch)
                else:
                    out_ch = ch
                    self.input_blocks.append(
                        TimestepEmbedSequential(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                            )
                            if resblock_updown
                            else Downsample(
                                ch, conv_resample, dims=dims, out_channels=out_ch
                            )
                        )
                    )
                    ch = out_ch
                    input_block_chans.append(ch)
                    ds *= 2
                    self._feature_size += ch

        if not self.use_classifier_free_diffusion:
            self.middle_block = TimestepEmbedSequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    use_new_attention_order=use_new_attention_order,
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            )
            self._feature_size += ch
        else:
            self.middle_block = EmbedSequential(
                ClassifierFreeResBlock(ch, ch, time_embed_dim, time_embed_dim, self.dropout),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=num_head_channels,
                    use_new_attention_order=use_new_attention_order,
                ),
                ClassifierFreeResBlock(ch, ch, time_embed_dim, time_embed_dim, self.dropout)
            )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                if not self.use_classifier_free_diffusion:
                    layers = [
                        ResBlock(
                            ch + ich,
                            time_embed_dim,
                            dropout,
                            out_channels=int(model_channels * mult),
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    ]
                else:
                    next_channel = model_channels * mult
                    layers = [
                        ClassifierFreeResBlock(ch+ich, next_channel, time_embed_dim, time_embed_dim, self.dropout)
                    ]
                ch = int(model_channels * mult)
                if not self.use_classifier_free_diffusion:
                    if ds in attention_resolutions:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                            )
                        )
                else:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    if not self.use_classifier_free_diffusion:
                        layers.append(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                            if resblock_updown
                            else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                        ds //= 2
                    else:
                        layers.append(
                            ClassifierFreeUpsample(ch, ch)
                        )
                if not self.use_classifier_free_diffusion:
                    self.output_blocks.append(TimestepEmbedSequential(*layers))
                else:
                    self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def execute(self, x, timesteps, y=None, cemb=jt.array(0)):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
            
        if self.use_classifier_free_diffusion:
            cemb = self.cemb_layer(cemb)
            hs = []
            h = x.unary(self.dtype)
            for block in self.input_blocks:
                h = block(h, emb, cemb)
                hs.append(h)
            h = self.middle_block(h, emb, cemb)
            for block in self.output_blocks:
                h = jt.concat([h, hs.pop()], dim = 1)
                h = block(h, emb, cemb)
            h = h.unary(self.dtype)
        else:
            h = x.unary(self.dtype)
            # print("out: ", h.shape)
            for module in self.input_blocks:
                h = module(h, emb)
                hs.append(h)
                # print("in: ", h.shape)
            h = self.middle_block(h, emb)
            for module in self.output_blocks:
                h = jt.concat([h, hs.pop()], dim=1)
                h = module(h, emb)
            h = h.unary(x.dtype)
        
        return self.out(h)


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def execute(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = nn.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = jt.concat([x, upsampled], dim=1)
        return super().execute(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = jt.float16 if use_fp16 else jt.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def execute(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.unary(op=self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.unary(x.dtype).mean(dims=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.unary(x.dtype).mean(dims=(2, 3)))
            h = jt.concat(results, axis=-1)
            return self.out(h)
        else:
            h = h.unary(x.dtype)
            return self.out(h)
        
from abc import abstractmethod
import math

def timestep_embedding(timesteps:jt.Var, dim:int, max_period=10000) -> jt.Var:
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
    embedding = jt.cat([jt.cos(args), jt.sin(args)], dim=-1)
    if dim % 2:
        embedding = jt.cat([embedding, jt.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ClassifierFreeUpsample(nn.Module):
    """
    an upsampling layer
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.layer = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1)
    def execute(self, x:jt.Var) -> jt.Var:
        assert x.shape[1] == self.in_ch, f'x and upsampling layer({self.in_ch}->{self.out_ch}) doesn\'t match.'
        x = nn.interpolate(x, scale_factor = 2, mode = "nearest")
        output = self.layer(x)
        return output

class ClassifierFreeDownsample(nn.Module):
    """
    a downsampling layer
    """
    def __init__(self, in_ch:int, out_ch:int, use_conv:bool):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        if use_conv:
            self.layer = nn.Conv2d(self.in_ch, self.out_ch, kernel_size = 3, stride = 2, padding = 1)
        else:
            self.layer = nn.AvgPool2d(kernel_size = 2, stride = 2)
    def execute(self, x:jt.Var) -> jt.Var:
        assert x.shape[1] == self.in_ch, f'x and upsampling layer({self.in_ch}->{self.out_ch}) doesn\'t match.'
        return self.layer(x)


class AttnBlock(nn.Module):
    def __init__(self, in_ch:int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)

    def execute(self, x:jt.Var) -> jt.Var:
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = jt.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = nn.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = jt.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h

class Unet(nn.Module):
    def __init__(self, in_ch=3, mod_ch=64, out_ch=3, ch_mul=[1,2,4,8], num_res_blocks=2, cdim=10, use_conv=True, droprate=0, dtype=jt.float32):
        super().__init__()
        self.in_ch = in_ch
        self.mod_ch = mod_ch
        self.out_ch = out_ch
        self.ch_mul = ch_mul
        self.num_res_blocks = num_res_blocks
        self.cdim = cdim
        self.use_conv = use_conv
        self.droprate = droprate
        # self.num_heads = num_heads
        self.dtype = dtype
        tdim = mod_ch * 4
        self.temb_layer = nn.Sequential(
            nn.Linear(mod_ch, tdim),
            SiLU(),
            nn.Linear(tdim, tdim),
        )
        self.cemb_layer = nn.Sequential(
            nn.Linear(self.cdim, tdim),
            SiLU(),
            nn.Linear(tdim, tdim),
        )
        self.downblocks = nn.ModuleList([
            EmbedSequential(nn.Conv2d(in_ch, self.mod_ch, 3, padding=1))
        ])
        now_ch = self.ch_mul[0] * self.mod_ch
        chs = [now_ch]
        for i, mul in enumerate(self.ch_mul):
            nxt_ch = mul * self.mod_ch
            for _ in range(self.num_res_blocks):
                layers = [
                    ClassifierFreeResBlock(now_ch, nxt_ch, tdim, tdim, self.droprate),
                    AttnBlock(nxt_ch)
                ]
                now_ch = nxt_ch
                self.downblocks.append(EmbedSequential(*layers))
                chs.append(now_ch)
            if i != len(self.ch_mul) - 1:
                self.downblocks.append(EmbedSequential(ClassifierFreeDownsample(now_ch, now_ch, self.use_conv)))
                chs.append(now_ch)
        self.middleblocks = EmbedSequential(
            ClassifierFreeResBlock(now_ch, now_ch, tdim, tdim, self.droprate),
            AttnBlock(now_ch),
            ClassifierFreeResBlock(now_ch, now_ch, tdim, tdim, self.droprate)
        )
        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(self.ch_mul))[::-1]:
            nxt_ch = mul * self.mod_ch
            for j in range(num_res_blocks + 1):
                layers = [
                    ClassifierFreeResBlock(now_ch+chs.pop(), nxt_ch, tdim, tdim, self.droprate),
                    AttnBlock(nxt_ch)
                ]
                now_ch = nxt_ch
                if i and j == self.num_res_blocks:
                    layers.append(ClassifierFreeUpsample(now_ch, now_ch))
                self.upblocks.append(EmbedSequential(*layers))
        self.out = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            SiLU(),
            nn.Conv2d(now_ch, self.out_ch, 3, stride = 1, padding = 1)
        )
    def execute(self, x:jt.Var, t:jt.Var, cemb:jt.Var) -> jt.Var:
        temb = self.temb_layer(timestep_embedding(t, self.mod_ch))
        cemb = self.cemb_layer(cemb)
        hs = []
        h = x.unary(self.dtype)
        for block in self.downblocks:
            h = block(h, temb, cemb)
            hs.append(h)
        h = self.middleblocks(h, temb, cemb)
        for block in self.upblocks:
            h = jt.cat([h, hs.pop()], dim = 1)
            h = block(h, temb, cemb)
        h = h.unary(self.dtype)
        return self.out(h)
