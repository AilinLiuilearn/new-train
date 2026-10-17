# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner import CheckpointLoader
from mmengine.utils import to_2tuple

from mmseg.registry import MODELS
from semseg.models.backbones.embed import PatchEmbed, PatchMerging


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape, submode):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape, submode):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x

import math
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.0625, act_layer=nn.ReLU, skip_connect=True, prompt_add=False):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.D_fc1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.D_fc2.weight)
            nn.init.zeros_(self.D_fc1.bias)
            nn.init.zeros_(self.D_fc2.bias)
        self.prompt_add = prompt_add
        if prompt_add:
            self.D_fc_prompt = nn.Linear(D_features, D_hidden_features)

    def forward(self, x, prompt=None):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        if self.prompt_add and prompt is not None:
            prompts = self.D_fc_prompt(prompt)
            xs = xs + prompts
        xs = self.act(xs)
        xs = nn.functional.dropout(xs, p=0.1, training=self.training)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

#MAPA-Adaptation
class SwinBlockAdapter(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None,
                 adapter_ratio=0.0625):

        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

        self.MLP_RGB_Adapter = Adapter(embed_dims, mlp_ratio=adapter_ratio, skip_connect=False)
        self.MLP_DTE_Adapter = Adapter(embed_dims, mlp_ratio=adapter_ratio, skip_connect=False)
        self.scale = 0.5
        self.adapter_dict = {
            'rgb': {'mlp': self.MLP_RGB_Adapter},
            'dte': {'mlp': self.MLP_DTE_Adapter},
        }


    def forward(self, x, hw_shape, sub_mode):

        def _inner_forward(x, sub_mode):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            #adapter
            adapter_x = self.scale * self.adapter_dict[sub_mode]['mlp'](x)

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            # adapter
            x = x + adapter_x

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, sub_mode)
        else:
            x = _inner_forward(x, sub_mode)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None,
                 adapter_ratio=0.0625):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlockAdapter(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None,
                adapter_ratio=adapter_ratio)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape, sub_mode):
        for block in self.blocks:
            x = block(x, hw_shape, sub_mode)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape, sub_mode)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape

import einops
from timm.models.layers import DropPath
class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')



class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

# Follow DAT
class DAttentionMM(nn.Module):

    def __init__(
            self, dims, q_size=(60, 80), kv_size=56, n_heads=2, n_groups=1,
            attn_drop=0, proj_drop=0, stride=8,
            offset_range_factor=-1, use_pe=True, dwc_pe=False,
            no_off=False, fixed_pe=False, ksize=9, log_cpb=False, dpr=0, level=None
    ):
        super().__init__()
        self.fp16_enabled = False
        self.drop_path = DropPath(dpr)
        n_head_channels = dims // n_heads
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.stride = stride
        self.log_cpb = log_cpb
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset_x = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        self.conv_offset_y = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.fuse_q = conv_bn_relu(int(dims * 2), dims)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.get_sample_weight = nn.Sequential(
            nn.Conv2d(
                dims, dims,
                kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                dims, 2,
                kernel_size=1, stride=1, padding=0
            ),
        )
        self.softmax = nn.Softmax(dim=1)

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        level_weight = [1e-3, 1e-3, 1e-3, 1]

        self.deform_weight = nn.Parameter(level_weight[level] * torch.ones((dims)), requires_grad=True)
        self.identity_weight = nn.Parameter(torch.ones((dims)), requires_grad=True)

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and (not self.no_off):
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        with torch.no_grad():
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
                torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
                # indexing='ij'
            )
            ref = torch.stack((ref_y, ref_x), -1)
            ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
            ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
            ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

            return ref

    def _get_q_grid(self, H, W, B, dtype, device):
        with torch.no_grad():
            ref_y, ref_x = torch.meshgrid(
                torch.arange(0, H, dtype=dtype, device=device),
                torch.arange(0, W, dtype=dtype, device=device),
                # indexing='ij'
            )
            ref = torch.stack((ref_y, ref_x), -1)
            ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
            ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
            ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

            return ref

    def forward(self, x, y):
        # x-rgb y-dte
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        xy = torch.cat([x, y], dim=1)

        xy = self.fuse_q(xy)
        q = self.proj_q(xy)
        x_off = einops.rearrange(x, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        y_off = einops.rearrange(y, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)

        x_offset = self.conv_offset_x(x_off).contiguous()  # B * g 2 Hg Wg
        y_offset = self.conv_offset_y(y_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = x_offset.size(2), x_offset.size(3)

        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        x_offset = einops.rearrange(x_offset, 'b p h w -> b h w p')
        y_offset = einops.rearrange(y_offset, 'b p h w -> b h w p')

        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos_x = (x_offset + reference).clamp(-1., +1.)
            pos_y = (y_offset + reference).clamp(-1., +1.)
            # pos_x = (x_offset + reference).tanh()
            # pos_y = (y_offset + reference).tanh()

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            Hk, Wk = x_sampled.size(2), x_sampled.size(3)
            n_sample = Hk * Wk
        else:
            x_sampled_x = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos_x[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
            x_sampled_y = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos_y[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
            y_sampled_x = F.grid_sample(
                input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos_x[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
            y_sampled_y = F.grid_sample(
                input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos_y[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
            q_sampled_x = F.grid_sample(
                input=q.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos_x[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
            q_sampled_y = F.grid_sample(
                input=q.reshape(B * self.n_groups, self.n_group_channels, H, W),
                grid=pos_y[..., (1, 0)],  # y, x -> x, y
                mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg

        x_sampled_x = x_sampled_x.reshape(B, C, 1, n_sample)
        x_sampled_y = x_sampled_y.reshape(B, C, 1, n_sample)
        x_sampled = torch.cat([x_sampled_x, x_sampled_y], dim=-1)
        y_sampled_x = y_sampled_x.reshape(B, C, 1, n_sample)
        y_sampled_y = y_sampled_y.reshape(B, C, 1, n_sample)
        y_sampled = torch.cat([y_sampled_x, y_sampled_y], dim=-1)
        q_sampled_x = q_sampled_x.reshape(B, C, 1, n_sample)
        q_sampled_y = q_sampled_y.reshape(B, C, 1, n_sample)
        q_sampled = torch.cat([q_sampled_x, q_sampled_y], dim=-1)

        sample_weight = self.get_sample_weight(q_sampled)
        sample_weight = self.softmax(sample_weight).squeeze(2).unsqueeze(1)
        sampled = torch.sum(sample_weight * torch.cat([x_sampled, y_sampled], dim=-2), dim=-2, keepdim=True)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample * 2)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample * 2)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels,
                                                                              H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn_bias = F.interpolate(attn_bias, size=(H * W, n_sample), mode='bilinear', align_corners=True)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (
                        q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups,
                                                                                               n_sample,
                                                                                               2).unsqueeze(1)).mul(
                    4.0)  # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement)  # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)

                q_grid = self._get_q_grid(H, W, B, dtype, device)

                displacement_x = (
                        q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos_x.reshape(B * self.n_groups,
                                                                                                 n_sample,
                                                                                                 2).unsqueeze(1)).mul(
                    0.5)

                displacement_y = (
                        q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos_y.reshape(B * self.n_groups,
                                                                                                 n_sample,
                                                                                                 2).unsqueeze(1)).mul(
                    0.5)

                attn_bias_x = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement_x[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns

                attn_bias_y = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads,
                                           g=self.n_groups),
                    grid=displacement_y[..., (1, 0)],
                    mode='bilinear', align_corners=True
                )  # B * g, h_g, HW, Ns
                attn_bias = torch.cat([attn_bias_x, attn_bias_y], dim=-1)
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample * 2)

                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        out = self.proj_drop(self.proj_out(out))

        out = self.deform_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * out + self.identity_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * xy
        return out


def init_tfts(dim):
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=.02)
    nn.init.normal_(beta, std=.02)
    return gamma, beta

def apply_tfts(x, gamma, beta):
    assert gamma.shape == beta.shape
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    elif x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1) + beta.view(1, -1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')

#MAPA-Prompting
class MPGBlock(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        self.D_fc1 = nn.Linear(dim, int(dim * ratio))
        self.D_fc2 = nn.Linear(dim, int(dim * ratio))
        self.P_fc2 = nn.Linear(int(dim * ratio * 2), int(dim * ratio))
        self.U_fc1 = nn.Linear(int(dim * ratio), dim)
        self.act = nn.GELU()
        self.tfts_gamma_rgb, self.tfts_beta_rgb = init_tfts(dim)
        self.tfts_gamma_dte, self.tfts_beta_dte = init_tfts(dim)

    def forward(self, x_rgb, x_dte, H, W):
        # Unifying Dual-modality Features
        x_rgb = self.D_fc1(x_rgb)
        x_dte = self.D_fc2(x_dte)
        x = torch.cat([x_rgb, x_dte], dim=-1)
        x = self.P_fc2(x)
        x = self.U_fc1(x)

        # Prompt Generation
        p_rgb = apply_tfts(x, self.tfts_gamma_rgb, self.tfts_beta_rgb)
        p_dte = apply_tfts(x, self.tfts_gamma_dte, self.tfts_beta_dte)

        return x + p_rgb, x + p_dte

# DSCF
class DeformMPGBlock(nn.Module):
    def __init__(self, dims, stride, n_groups, n_heads, dpr, level, ratio):
        super().__init__()
        #self.patch_embed = PatchEmbed(dim, dim, 3, 1, 1)
        self.D_fc1 = nn.Linear(dims, int(dims * ratio))
        self.D_fc2 = nn.Linear(dims, int(dims * ratio))
        self.U_fc1 = nn.Linear(int(dims * ratio), dims)
        self.act = nn.GELU()
        self.deform_atten = DAttentionMM(dims = int(dims * ratio), stride = stride, n_groups = n_groups, n_heads = n_heads, dpr=dpr,level = level)


    def forward(self, x_rgb, x_dte, H, W, level):
        x_rgb = self.D_fc1(x_rgb)
        x_dte = self.D_fc2(x_dte)
        B, N, new_C = x_rgb.shape
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_dte = x_dte.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_fuse = self.deform_atten(x_rgb, x_dte)
        x_fuse = x_fuse.reshape(B, new_C, -1).permute(0, 2, 1).contiguous()
        x_fuse = self.U_fc1(x_fuse)
        return x_fuse

import random
def apply_mask(rgb, dte):
    batch_size = rgb.size(0)

    # Half of the image indices are randomly selected for masking, and the other half are computed normally.
    indices_to_mask = random.sample(range(batch_size), batch_size // 2)

    # The following two lines of code are only applicable when the batch_size is 4;
    # for other batch_sizes, only minor modifications to these codes are needed.
    rgb[indices_to_mask[0]] = torch.zeros_like(rgb[indices_to_mask[0]])
    dte[indices_to_mask[1]] = torch.zeros_like(dte[indices_to_mask[1]])

    return rgb, dte


checkpoint_file = '/mnt/csip-107/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'
# checkpoint_file = '/mnt/csip-107/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
@MODELS.register_module()
class SwinTransformer(BaseModule):
    """Swin Transformer backbone.

    This backbone is the implementation of `Swin Transformer:
    Hierarchical Vision Transformer using Shifted
    Windows <https://arxiv.org/abs/2103.14030>`_.
    Inspiration from https://github.com/microsoft/Swin-Transformer.

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int | float): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LN').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 pretrain_img_size=384,
                 in_channels=3,
                 embed_dims=128,
                 patch_size=4,
                 window_size=12,
                 mlp_ratio=4,
                 depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
                 mapa_ratio = 0.125,
                 adapter_ratio = 0.0625,
                 dscf_ratio = 0.125,
                 ):
        self.frozen_stages = frozen_stages

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super().__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        # dte extra patch_embed
        self.extra_patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]


        # ours
        self.stages = ModuleList()
        self.MPGBlocks = ModuleList()
        self.DeformMPGBlocks = ModuleList()

        stride = [8, 4, 2, 1]
        n_groups = [1, 2, 4, 8]
        n_heads = [2, 4, 8, 16]
        in_channels = embed_dims


        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None
            # MAPA
            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_channels),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None,
                adapter_ratio=adapter_ratio)
            mpg = MPGBlock(in_channels, mapa_ratio)
            #DCSF
            dmb = DeformMPGBlock(dims=in_channels, stride=stride[i],
                                                  n_groups=n_groups[i], n_heads=n_heads[i],
                                                  dpr=0, level=i, ratio=dscf_ratio)
            self.MPGBlocks.append(mpg)
            self.stages.append(stage)
            self.DeformMPGBlocks.append(dmb)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
            dte_layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            dte_layer_name = f'extra_norm{i}'
            self.add_module(dte_layer_name, dte_layer)
            deform_layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            deform_layer_name = f'fuse_norm{i}'
            self.add_module(deform_layer_name, deform_layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    print_log('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                if table_key in self.state_dict():
                    table_current = self.state_dict()[table_key]
                    L1, nH1 = table_pretrained.size()
                    L2, nH2 = table_current.size()
                    if nH1 != nH2:
                        print_log(f'Error in loading {table_key}, pass')
                    elif L1 != L2:
                        S1 = int(L1**0.5)
                        S2 = int(L2**0.5)
                        table_pretrained_resized = F.interpolate(
                            table_pretrained.permute(1, 0).reshape(
                                1, nH1, S1, S1),
                            size=(S2, S2),
                            mode='bicubic')
                        state_dict[table_key] = table_pretrained_resized.view(
                            nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # rgb and dte
        x_rgb = x[0]
        x_dte = x[1]

        # patch_embed
        x_rgb, hw_shape_rgb = self.patch_embed(x_rgb)
        x_dte, hw_shape_ext = self.extra_patch_embed(x_dte)

        # MMST
        if self.training:
            x_rgb, x_dte = apply_mask(x_rgb, x_dte)


        outs = []
        outs_rgb = []
        outs_dte = []

        for i, stage in enumerate(self.stages):
            # MAPA
            x_rgb_fuse, x_dte_fuse = self.MPGBlocks[i](x_rgb, x_dte, hw_shape_rgb[0], hw_shape_rgb[1])
            x_rgb = x_rgb + x_rgb_fuse
            x_dte = x_dte + x_dte_fuse
            x_rgb, hw_shape_rgb, x_rgb_out, out_hw_shape = stage(x_rgb, hw_shape_rgb, sub_mode='rgb')
            x_dte, hw_shape_ext, x_dte_out, out_hw_shape = stage(x_dte, hw_shape_ext, sub_mode='dte')

            # out
            if i in self.out_indices:
                # rgb
                rgb_norm_layer = getattr(self, f'norm{i}')
                x_rgb_out = rgb_norm_layer(x_rgb_out)

                # dte
                dte_norm_layer = getattr(self, f'extra_norm{i}')
                x_dte_out = dte_norm_layer(x_dte_out)

                # fusion
                out = self.DeformMPGBlocks[i](x_rgb_out, x_dte_out, *out_hw_shape, i)
                fuse_norm_layer = getattr(self, f'fuse_norm{i}')
                out = fuse_norm_layer(out)

                # reshape
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                x_rgb_out = x_rgb_out.view(-1, *out_hw_shape,
                                           self.num_features[i]).permute(0, 3, 1,
                                                                         2).contiguous()
                x_dte_out = x_dte_out.view(-1, *out_hw_shape,
                                           self.num_features[i]).permute(0, 3, 1,
                                                                         2).contiguous()

                outs.append(out)
                outs_rgb.append(x_rgb_out)
                outs_dte.append(x_dte_out)

        return outs, outs_rgb, outs_dte

if __name__ == '__main__':
    modals = ['img', 'depth', 'event', 'lidar']
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    model = SwinTransformer()
    outs = model(torch.zeros(1, 3, 1024, 1024))
    for y in outs:
        print(y.shape)