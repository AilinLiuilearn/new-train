import torch
import torch.nn as nn
import torch.nn.functional as F


PET_LAP_HGL_LOG_KEYS = [
    'pet_beta',
    'pet_spatial_mean',
    'pet_spatial_std',
    'pet_modulation_delta_abs_mean',
    'down1_alpha',
    'down2_alpha',
    'down3_alpha',
    'down1_x_hp_abs_mean',
    'down1_x_lap_abs_mean',
    'down1_x_hh_mean',
    'down1_high_gate_mean',
    'down2_x_hp_abs_mean',
    'down2_x_lap_abs_mean',
    'down2_x_hh_mean',
    'down2_high_gate_mean',
    'down3_x_hp_abs_mean',
    'down3_x_lap_abs_mean',
    'down3_x_hh_mean',
    'down3_high_gate_mean',
]


def get_gn_groups(channels: int, max_groups: int = 8) -> int:
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        bias: bool = False,
        groups: int = 1,
        gn_groups: int = 8,
        act: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
        self.norm = nn.GroupNorm(get_gn_groups(out_ch, gn_groups), out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock2D(nn.Module):
    def __init__(self, channels: int, gn_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(channels, channels, kernel_size=3, gn_groups=gn_groups),
            ConvGNAct(channels, channels, kernel_size=3, gn_groups=gn_groups, act=False),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class FixedLaplacianHighPass(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        kernel = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.channels = channels

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight,
            bias=None,
            stride=1,
            padding=1,
            groups=self.channels,
        )


class LapHGLDownBlockLite(nn.Module):
    """Lightweight LapHGL down block: single conv branches, no ResBlock."""

    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 4):
        super().__init__()
        self.high_pass = FixedLaplacianHighPass(in_ch)
        self.hh_merge = ConvGNAct(in_ch * 2, in_ch, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.low_proj = ConvGNAct(in_ch, out_ch, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.high_proj = ConvGNAct(in_ch, out_ch, kernel_size=3, padding=1, gn_groups=gn_groups)
        self.gate_proj = nn.Sequential(
            ConvGNAct(in_ch, out_ch, kernel_size=3, padding=1, gn_groups=gn_groups),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        b, c, h, w = x.shape
        x_ll = F.avg_pool2d(x, kernel_size=2, stride=2)
        x_ll_up = F.interpolate(x_ll, size=(h, w), mode='bilinear', align_corners=False)
        x_lap = x - x_ll_up
        x_hp = self.high_pass(x)
        x_hh = self.hh_merge(torch.cat([x_hp.abs(), x_lap.abs()], dim=1))
        x_hh_down = F.avg_pool2d(x_hh, kernel_size=2, stride=2)

        low_feat = self.low_proj(x_ll)
        high_feat = self.high_proj(x_hh_down)
        high_gate = self.gate_proj(x_hh_down)
        out = low_feat + self.alpha * high_gate * high_feat

        aux = {
            'alpha': self.alpha.detach(),
            'x_ll_mean': x_ll.detach().mean(),
            'x_ll_std': x_ll.detach().std(),
            'x_hp_abs_mean': x_hp.detach().abs().mean(),
            'x_hp_abs_std': x_hp.detach().abs().std(),
            'x_lap_abs_mean': x_lap.detach().abs().mean(),
            'x_lap_abs_std': x_lap.detach().abs().std(),
            'x_hh_mean': x_hh.detach().mean(),
            'x_hh_std': x_hh.detach().std(),
            'high_gate_mean': high_gate.detach().mean(),
            'high_gate_std': high_gate.detach().std(),
        }
        return out, aux


def _laplacian_hh_features(pet):
    """Fixed Laplacian high-frequency map: merge |Laplacian(pet)| and |pet - up(pool(pet))|."""
    h, w = pet.shape[-2:]
    x_ll = F.avg_pool2d(pet, kernel_size=2, stride=2)
    x_ll_up = F.interpolate(x_ll, size=(h, w), mode='bilinear', align_corners=False)
    x_lap = pet - x_ll_up
    kernel = pet.new_tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]).view(1, 1, 3, 3)
    x_hp = F.conv2d(pet, kernel, padding=1)
    x_hh = torch.cat([x_hp.abs(), x_lap.abs()], dim=1)
    return x_hp, x_lap, x_hh


def _pack_lap_hgl_aux(s_pet, down_aux_list):
    aux = {
        'pet_spatial_mean': s_pet.detach().mean(),
        'pet_spatial_std': s_pet.detach().std(),
    }
    for idx, block_aux in enumerate(down_aux_list, start=1):
        prefix = f'down{idx}_'
        for key, value in block_aux.items():
            aux[prefix + key] = value
    for idx in range(len(down_aux_list) + 1, 4):
        aux[f'down{idx}_alpha'] = s_pet.new_tensor(0.0)
        aux[f'down{idx}_x_hp_abs_mean'] = s_pet.new_tensor(0.0)
        aux[f'down{idx}_x_lap_abs_mean'] = s_pet.new_tensor(0.0)
        aux[f'down{idx}_x_hh_mean'] = s_pet.new_tensor(0.0)
        aux[f'down{idx}_high_gate_mean'] = s_pet.new_tensor(0.0)
    return aux


class HighFreqLapHGLPrior(nn.Module):
    """
    Deprecated minimal path (high-freq only, single conv head).
    Kept for ablation via pet_prior_size=minimal.
    """

    def __init__(self, mid_channels: int = 32, gn_groups: int = 4):
        super().__init__()
        self.hh_stem = ConvGNAct(2, mid_channels, kernel_size=3, gn_groups=gn_groups)
        self.down_blocks = nn.ModuleList([
            ConvGNAct(mid_channels, mid_channels, kernel_size=3, stride=2, gn_groups=gn_groups)
            for _ in range(5)
        ])
        self.spatial_head = nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)

    def forward(self, pet, target_size=None):
        x_hp, x_lap, x_hh = _laplacian_hh_features(pet)
        feat = self.hh_stem(x_hh)
        for block in self.down_blocks:
            feat = block(feat)
        if target_size is not None and feat.shape[-2:] != target_size:
            feat = F.adaptive_avg_pool2d(feat, target_size)
        s_pet = torch.sigmoid(self.spatial_head(feat))

        block_aux = {
            'alpha': self.spatial_head.weight.new_tensor(0.0),
            'x_ll_mean': pet.detach().mean(),
            'x_ll_std': pet.detach().std(),
            'x_hp_abs_mean': x_hp.detach().abs().mean(),
            'x_hp_abs_std': x_hp.detach().abs().std(),
            'x_lap_abs_mean': x_lap.detach().abs().mean(),
            'x_lap_abs_std': x_lap.detach().abs().std(),
            'x_hh_mean': x_hh.detach().mean(),
            'x_hh_std': x_hh.detach().std(),
            'high_gate_mean': s_pet.detach().mean(),
            'high_gate_std': s_pet.detach().std(),
        }
        aux = _pack_lap_hgl_aux(s_pet, [block_aux])
        return s_pet, feat, aux


def _coarse_low_freq(pet, target_size):
    """Pyramid-style coarse LL: repeated avg-pool then resize to target."""
    ll = pet
    while ll.shape[-1] > max(target_size[1], 8) or ll.shape[-2] > max(target_size[0], 8):
        ll = F.avg_pool2d(ll, kernel_size=2, stride=2)
    if ll.shape[-2:] != target_size:
        ll = F.interpolate(ll, size=target_size, mode='bilinear', align_corners=False)
    return ll


class PETLapHGLPriorEncoderLite(nn.Module):
    """
    Full 4-stage LapHGL prior (lightweight).

    Same stage layout as the full encoder:
      stem -> F1; 3x LapHGLDown -> F2/F3/F4; multi-scale fusion.

    Lightweight via narrow channels + LapHGLDownBlockLite (no ResBlock).

    Low and high frequency both contribute to S_pet:
      - each down block: out = low_feat(X_ll) + alpha * gate * high_feat(X_hh)
      - final head: fuse multi-scale features + explicit LL map + explicit HH map
    """

    def __init__(
        self,
        in_ch: int = 1,
        channels=(24, 32, 48, 64),
        fuse_mid_channels: int = 32,
        spatial_channels: int = 64,
        gn_groups: int = 4,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = ConvGNAct(in_ch, c1, kernel_size=7, stride=4, padding=3, gn_groups=gn_groups)
        self.down1 = LapHGLDownBlockLite(c1, c2, gn_groups=gn_groups)
        self.down2 = LapHGLDownBlockLite(c2, c3, gn_groups=gn_groups)
        self.down3 = LapHGLDownBlockLite(c3, c4, gn_groups=gn_groups)

        self.proj1 = ConvGNAct(c1, fuse_mid_channels, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.proj2 = ConvGNAct(c2, fuse_mid_channels, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.proj3 = ConvGNAct(c3, fuse_mid_channels, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.proj4 = ConvGNAct(c4, fuse_mid_channels, kernel_size=1, padding=0, gn_groups=gn_groups)

        self.ms_fuse = ConvGNAct(
            fuse_mid_channels * 4, spatial_channels, kernel_size=1, padding=0, gn_groups=gn_groups,
        )
        self.low_ll_proj = ConvGNAct(in_ch, fuse_mid_channels, kernel_size=3, padding=1, gn_groups=gn_groups)
        self.high_hh_proj = ConvGNAct(2, fuse_mid_channels, kernel_size=3, padding=1, gn_groups=gn_groups)
        self.spatial_fuse = ConvGNAct(
            spatial_channels + fuse_mid_channels * 2,
            spatial_channels,
            kernel_size=3,
            padding=1,
            gn_groups=gn_groups,
        )
        self.spatial_head = nn.Conv2d(spatial_channels, 1, kernel_size=1, bias=True)

    def forward(self, pet, target_size=None):
        f1 = self.stem(pet)
        f2, aux1 = self.down1(f1)
        f3, aux2 = self.down2(f2)
        f4, aux3 = self.down3(f3)

        if target_size is None:
            target_size = f4.shape[-2:]

        f1_t = self.proj1(F.adaptive_avg_pool2d(f1, target_size))
        f2_t = self.proj2(F.adaptive_avg_pool2d(f2, target_size))
        f3_t = self.proj3(F.adaptive_avg_pool2d(f3, target_size))
        f4_t = self.proj4(F.adaptive_avg_pool2d(f4, target_size))
        ms_feat = self.ms_fuse(torch.cat([f1_t, f2_t, f3_t, f4_t], dim=1))

        ll_map = _coarse_low_freq(pet, target_size)
        _, _, x_hh = _laplacian_hh_features(pet)
        hh_map = F.adaptive_avg_pool2d(x_hh, target_size)

        ll_feat = self.low_ll_proj(ll_map)
        hh_feat = self.high_hh_proj(hh_map)
        k_pet = self.spatial_fuse(torch.cat([ms_feat, ll_feat, hh_feat], dim=1))
        s_pet = torch.sigmoid(self.spatial_head(k_pet))

        aux = _pack_lap_hgl_aux(s_pet, [aux1, aux2, aux3])
        aux['pet_ll_mean'] = ll_map.detach().mean()
        aux['pet_ll_std'] = ll_map.detach().std()
        aux['pet_hh_mean'] = hh_map.detach().mean()
        aux['pet_hh_std'] = hh_map.detach().std()
        return s_pet, k_pet, aux


class LapHGLDownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.high_pass = FixedLaplacianHighPass(in_ch)
        self.hh_merge = nn.Sequential(
            ConvGNAct(in_ch * 2, in_ch, kernel_size=1, padding=0, gn_groups=gn_groups),
            ConvGNAct(in_ch, in_ch, kernel_size=3, padding=1, gn_groups=gn_groups),
        )
        self.low_proj = nn.Sequential(
            ConvGNAct(in_ch, out_ch, kernel_size=1, padding=0, gn_groups=gn_groups),
            ResBlock2D(out_ch, gn_groups=gn_groups),
        )
        self.high_proj = nn.Sequential(
            ConvGNAct(in_ch, out_ch, kernel_size=3, padding=1, gn_groups=gn_groups),
            ResBlock2D(out_ch, gn_groups=gn_groups),
        )
        self.gate_proj = nn.Sequential(
            ConvGNAct(in_ch, out_ch, kernel_size=3, padding=1, gn_groups=gn_groups),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = ResBlock2D(out_ch, gn_groups=gn_groups)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        b, c, h, w = x.shape
        x_ll = F.avg_pool2d(x, kernel_size=2, stride=2)
        x_ll_up = F.interpolate(x_ll, size=(h, w), mode='bilinear', align_corners=False)
        x_lap = x - x_ll_up
        x_hp = self.high_pass(x)
        x_hh = self.hh_merge(torch.cat([x_hp.abs(), x_lap.abs()], dim=1))
        x_hh_down = F.avg_pool2d(x_hh, kernel_size=2, stride=2)

        low_feat = self.low_proj(x_ll)
        high_feat = self.high_proj(x_hh_down)
        high_gate = self.gate_proj(x_hh_down)
        out = low_feat + self.alpha * high_gate * high_feat
        out = self.fuse(out)

        aux = {
            'alpha': self.alpha.detach(),
            'x_ll_mean': x_ll.detach().mean(),
            'x_ll_std': x_ll.detach().std(),
            'x_hp_abs_mean': x_hp.detach().abs().mean(),
            'x_hp_abs_std': x_hp.detach().abs().std(),
            'x_lap_abs_mean': x_lap.detach().abs().mean(),
            'x_lap_abs_std': x_lap.detach().abs().std(),
            'x_hh_mean': x_hh.detach().mean(),
            'x_hh_std': x_hh.detach().std(),
            'high_gate_mean': high_gate.detach().mean(),
            'high_gate_std': high_gate.detach().std(),
        }
        return out, aux


class PETLapHGLPriorEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        channels=(64, 128, 256, 512),
        c4_channels: int = 512,
        fuse_mid_channels: int = 128,
        gn_groups: int = 8,
    ):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = nn.Sequential(
            ConvGNAct(in_ch, c1, kernel_size=7, stride=4, padding=3, gn_groups=gn_groups),
            ResBlock2D(c1, gn_groups=gn_groups),
        )
        self.down1 = LapHGLDownBlock(c1, c2, gn_groups=gn_groups)
        self.down2 = LapHGLDownBlock(c2, c3, gn_groups=gn_groups)
        self.down3 = LapHGLDownBlock(c3, c4, gn_groups=gn_groups)
        self.proj1 = ConvGNAct(c1, fuse_mid_channels, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.proj2 = ConvGNAct(c2, fuse_mid_channels, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.proj3 = ConvGNAct(c3, fuse_mid_channels, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.proj4 = ConvGNAct(c4, fuse_mid_channels, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.fuse = nn.Sequential(
            ConvGNAct(fuse_mid_channels * 4, c4_channels, kernel_size=1, padding=0, gn_groups=gn_groups),
            ResBlock2D(c4_channels, gn_groups=gn_groups),
        )
        self.spatial_head = nn.Conv2d(c4_channels, 1, kernel_size=1, bias=True)

    def forward(self, pet, target_size=None):
        f1 = self.stem(pet)
        f2, aux2 = self.down1(f1)
        f3, aux3 = self.down2(f2)
        f4, aux4 = self.down3(f3)

        if target_size is None:
            target_size = f4.shape[-2:]

        f1_t = self.proj1(F.adaptive_avg_pool2d(f1, target_size))
        f2_t = self.proj2(F.adaptive_avg_pool2d(f2, target_size))
        f3_t = self.proj3(F.adaptive_avg_pool2d(f3, target_size))
        f4_t = self.proj4(F.adaptive_avg_pool2d(f4, target_size))
        k_pet = self.fuse(torch.cat([f1_t, f2_t, f3_t, f4_t], dim=1))
        s_pet = torch.sigmoid(self.spatial_head(k_pet))

        aux = {
            'pet_spatial_mean': s_pet.detach().mean(),
            'pet_spatial_std': s_pet.detach().std(),
            'down1_alpha': aux2['alpha'],
            'down1_x_ll_mean': aux2['x_ll_mean'],
            'down1_x_ll_std': aux2['x_ll_std'],
            'down1_x_hp_abs_mean': aux2['x_hp_abs_mean'],
            'down1_x_hp_abs_std': aux2['x_hp_abs_std'],
            'down1_x_lap_abs_mean': aux2['x_lap_abs_mean'],
            'down1_x_lap_abs_std': aux2['x_lap_abs_std'],
            'down1_x_hh_mean': aux2['x_hh_mean'],
            'down1_x_hh_std': aux2['x_hh_std'],
            'down1_high_gate_mean': aux2['high_gate_mean'],
            'down1_high_gate_std': aux2['high_gate_std'],
            'down2_alpha': aux3['alpha'],
            'down2_x_ll_mean': aux3['x_ll_mean'],
            'down2_x_ll_std': aux3['x_ll_std'],
            'down2_x_hp_abs_mean': aux3['x_hp_abs_mean'],
            'down2_x_hp_abs_std': aux3['x_hp_abs_std'],
            'down2_x_lap_abs_mean': aux3['x_lap_abs_mean'],
            'down2_x_lap_abs_std': aux3['x_lap_abs_std'],
            'down2_x_hh_mean': aux3['x_hh_mean'],
            'down2_x_hh_std': aux3['x_hh_std'],
            'down2_high_gate_mean': aux3['high_gate_mean'],
            'down2_high_gate_std': aux3['high_gate_std'],
            'down3_alpha': aux4['alpha'],
            'down3_x_ll_mean': aux4['x_ll_mean'],
            'down3_x_ll_std': aux4['x_ll_std'],
            'down3_x_hp_abs_mean': aux4['x_hp_abs_mean'],
            'down3_x_hp_abs_std': aux4['x_hp_abs_std'],
            'down3_x_lap_abs_mean': aux4['x_lap_abs_mean'],
            'down3_x_lap_abs_std': aux4['x_lap_abs_std'],
            'down3_x_hh_mean': aux4['x_hh_mean'],
            'down3_x_hh_std': aux4['x_hh_std'],
            'down3_high_gate_mean': aux4['high_gate_mean'],
            'down3_high_gate_std': aux4['high_gate_std'],
        }
        return s_pet, k_pet, aux


def _default_lite_channels(pet_channels):
    if pet_channels and len(pet_channels) >= 4:
        vals = [int(x) for x in pet_channels[:4]]
        if max(vals) <= 64:
            return tuple(vals)
    return (24, 32, 48, 64)


def build_pet_lap_hgl_encoder(
    pet_prior_size='lite',
    mid_channels=32,
    pet_channels=(64, 128, 256, 512),
    c4_channels=512,
    fuse_mid_channels=32,
    gn_groups=4,
):
    size = str(pet_prior_size).lower()
    if size == 'full':
        return PETLapHGLPriorEncoder(
            in_ch=1,
            channels=tuple(int(x) for x in pet_channels[:4]),
            c4_channels=c4_channels,
            fuse_mid_channels=max(fuse_mid_channels, 128),
            gn_groups=gn_groups,
        )
    if size == 'minimal':
        return HighFreqLapHGLPrior(mid_channels=mid_channels, gn_groups=min(4, gn_groups))
    lite_channels = _default_lite_channels(pet_channels)
    return PETLapHGLPriorEncoderLite(
        in_ch=1,
        channels=lite_channels,
        fuse_mid_channels=min(fuse_mid_channels, 32),
        spatial_channels=min(c4_channels, 64),
        gn_groups=min(4, gn_groups),
    )


class DeepPETSpatialModulation(nn.Module):
    def __init__(
        self,
        c4_channels: int = 512,
        pet_channels=(64, 128, 256, 512),
        fuse_mid_channels: int = 128,
        gn_groups: int = 8,
        pet_prior_size: str = 'lite',
        pet_prior_mid_channels: int = 32,
    ):
        super().__init__()
        self.pet_prior_encoder = build_pet_lap_hgl_encoder(
            pet_prior_size=pet_prior_size,
            mid_channels=pet_prior_mid_channels,
            pet_channels=pet_channels,
            c4_channels=c4_channels,
            fuse_mid_channels=fuse_mid_channels,
            gn_groups=gn_groups,
        )
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, c4, pet):
        target_size = c4.shape[-2:]
        s_pet, k_pet, aux = self.pet_prior_encoder(pet, target_size=target_size)
        c4_enhanced = c4 * (1.0 + self.beta * s_pet)
        aux['pet_beta'] = self.beta.detach()
        aux['ct_c4_mean'] = c4.detach().mean()
        aux['ct_c4_std'] = c4.detach().std()
        aux['c4_enhanced_mean'] = c4_enhanced.detach().mean()
        aux['c4_enhanced_std'] = c4_enhanced.detach().std()
        aux['pet_modulation_delta_abs_mean'] = (c4_enhanced - c4).detach().abs().mean()
        return c4_enhanced, aux


class PETIntensitySpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def minmax_norm(x, eps: float = 1e-6):
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def forward(self, c4, pet):
        target_size = c4.shape[-2:]
        s_pet = F.interpolate(pet, size=target_size, mode='bilinear', align_corners=False)
        s_pet = self.minmax_norm(s_pet)
        c4_enhanced = c4 * (1.0 + self.beta * s_pet)
        aux = {
            'pet_beta': self.beta.detach(),
            'pet_spatial_mean': s_pet.detach().mean(),
            'pet_spatial_std': s_pet.detach().std(),
            'ct_c4_mean': c4.detach().mean(),
            'ct_c4_std': c4.detach().std(),
            'c4_enhanced_mean': c4_enhanced.detach().mean(),
            'c4_enhanced_std': c4_enhanced.detach().std(),
            'pet_modulation_delta_abs_mean': (c4_enhanced - c4).detach().abs().mean(),
        }
        return c4_enhanced, aux
