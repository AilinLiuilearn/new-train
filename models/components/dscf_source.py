"""Unchanged DSCF class bodies extracted from official imcjx/KTB.

Keep the Balance, CVPR 2025. Apache-2.0; see third_party/KTB/LICENSE.
Pinned commit: 75370dd584fb9d8571bced1aabade198dfbdc80f
Source: semseg/models/backbones/swin.py (blob e333b8f4f881b1f4846682d4470f305ac0158291).
Only unrelated Swin/MAPA classes and framework imports are omitted.
Integration changes live in dscf_residual.py, not in these class bodies.
The integration uses only the native default attention path:
no_off=False, offset_range_factor=-1, fixed_pe=False, dwc_pe=False,
log_cpb=False, n_groups=1; other upstream branches are not supported.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import einops
from timm.models.layers import DropPath, trunc_normal_

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
