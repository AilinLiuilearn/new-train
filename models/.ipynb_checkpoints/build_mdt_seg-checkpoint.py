import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.emcad_decoder import EMCADDecoder


def _unwrap_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ('state_dict', 'model', 'module'):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    return state_dict


def _sanitize_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ('module.', 'model.', 'backbone.', 'encoder.', 'visual.'):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        nk = nk.replace('stages.', 'stages_')
        cleaned[nk] = v
    return cleaned


def load_local_weights_safe(model, path, name='Encoder'):
    if not path or not os.path.exists(path):
        print(f'[-] {name}: Path not found {path}. Training from scratch.')
        return
    if os.path.isdir(path):
        for cand in ('pytorch_model.bin', 'model.safetensors', 'pvt_v2_b2.pth', 'pvt_v2_b0.pth'):
            full = os.path.join(path, cand)
            if os.path.exists(full):
                path = full
                break
    print(f'[+] {name}: Loading local weights from {path}')
    try:
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        state_dict = torch.load(path, map_location='cpu')
    state_dict = _sanitize_state_dict(_unwrap_state_dict(state_dict))

    model_state = model.state_dict()
    loadable = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped.append(k)
    msg = model.load_state_dict(loadable, strict=False)
    print(f'[+] {name} loaded params: {len(loadable)}, skipped: {len(skipped)}')
    print(f'[+] {name} load status: {msg}')


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class WeakFrequencyInjector(nn.Module):
    def __init__(self, channels, low_ratio=0.15, aux_prefix='x3', alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.freq_block = MedicalFrequencyFusionBlock(channels, low_ratio=low_ratio, aux_prefix=aux_prefix, enable_edge=False)

    def forward(self, feat_ct, feat_pet):
        base = feat_ct + feat_pet
        freq, aux = self.freq_block(feat_ct, feat_pet)
        out = (1.0 - self.alpha) * base + self.alpha * freq
        aux[f'{self.freq_block.aux_prefix}_base'] = base
        return out, aux


class ZeroInitX4Adapter(nn.Module):
    def __init__(self, channels, aux_prefix='x4'):
        super().__init__()
        hidden = max(channels // 4, 64)
        self.aux_prefix = aux_prefix
        self.pet_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.ct_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.mix = DepthwiseSeparableConv(channels, channels)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.res_scale = nn.Parameter(torch.zeros(1))
        self.edge_head = nn.Conv2d(channels, 1, 1)

    def forward(self, feat_ct, feat_pet):
        base = feat_ct + feat_pet
        pet_gate = self.pet_gate(feat_pet)
        ct_gate = self.ct_gate(feat_ct)
        guided = pet_gate * feat_ct + ct_gate * feat_pet
        mixed = self.mix(guided)
        refine = self.out_proj(mixed)
        out = base + self.res_scale.view(1, 1, 1, 1) * refine
        prefix = self.aux_prefix
        aux = {
            f'{prefix}_base': base,
            f'{prefix}_pet_gate': pet_gate,
            f'{prefix}_ct_gate': ct_gate,
            f'{prefix}_guided': guided,
            f'{prefix}_refine': refine,
            f'{prefix}_res_scale': self.res_scale.detach().view(1, 1, 1, 1),
            f'{prefix}_edge_logit': self.edge_head(base + refine),
        }
        return out, aux


class GatedWarmupX4Lite(nn.Module):
    def __init__(self, channels, low_ratio=0.15, aux_prefix='x4', warmup_epochs=5, gate_init=-8.0):
        super().__init__()
        self.aux_prefix = aux_prefix
        self.warmup_epochs = int(warmup_epochs)
        self.current_epoch = 0
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.freq_block = MedicalFrequencyFusionBlock(channels, low_ratio=low_ratio, aux_prefix=aux_prefix, enable_edge=True)

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _gate_value(self, ref_tensor):
        if self.current_epoch <= self.warmup_epochs:
            return ref_tensor.new_zeros(1, 1, 1, 1)
        return torch.sigmoid(self.gate_logit).view(1, 1, 1, 1).to(dtype=ref_tensor.dtype, device=ref_tensor.device)

    def forward(self, feat_ct, feat_pet):
        base = feat_ct + feat_pet
        lite, aux = self.freq_block(feat_ct, feat_pet)
        gate = self._gate_value(base)
        delta = lite - base
        out = base + gate * delta
        prefix = self.aux_prefix
        aux[f'{prefix}_base'] = base
        aux[f'{prefix}_lite'] = lite
        aux[f'{prefix}_delta'] = delta
        aux[f'{prefix}_gate'] = gate.expand(base.size(0), 1, 1, 1)
        return out, aux


class PLBUFusionLite(nn.Module):
    def __init__(self, channels, aux_prefix='x4', warmup_epochs=3, gate_init=-6.0, reduction=4,
                 enable_pet_reliability_gate=True, pet_reliability_floor=0.15, pet_reliability_scale=8.0):
        super().__init__()
        self.aux_prefix = aux_prefix
        self.warmup_epochs = int(warmup_epochs)
        self.current_epoch = 0
        hidden = max(channels // reduction, 32)

        self.pet_prior_head = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        self.ct_prior_head = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        self.ct_boundary_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        self.delta_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        nn.init.zeros_(self.delta_proj[-1].weight)
        nn.init.zeros_(self.delta_proj[-1].bias)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.enable_pet_reliability_gate = bool(enable_pet_reliability_gate)
        self.pet_reliability_floor = float(pet_reliability_floor)
        self.pet_reliability_scale = float(pet_reliability_scale)

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _warm_gate(self, ref_tensor):
        if self.current_epoch <= self.warmup_epochs:
            return ref_tensor.new_zeros(1, 1, 1, 1)
        return torch.sigmoid(self.gate_logit).view(1, 1, 1, 1).to(dtype=ref_tensor.dtype, device=ref_tensor.device)

    def forward(self, feat_ct, feat_pet):
        prefix = self.aux_prefix
        base = feat_ct + feat_pet

        lesion_logit_pet = self.pet_prior_head(feat_pet)
        lesion_logit_ct = self.ct_prior_head(feat_ct)
        lesion_prior_pet = torch.sigmoid(lesion_logit_pet)
        lesion_prior_ct = torch.sigmoid(lesion_logit_ct)

        if self.enable_pet_reliability_gate:
            pet_energy = feat_pet.abs().mean(dim=1, keepdim=True)
            ct_energy = feat_ct.abs().mean(dim=1, keepdim=True)
            rel_raw = torch.sigmoid((pet_energy - ct_energy) * self.pet_reliability_scale)
            pet_reliability = self.pet_reliability_floor + (1.0 - self.pet_reliability_floor) * rel_raw
        else:
            pet_reliability = torch.ones_like(lesion_prior_pet)

        lesion_prior = pet_reliability * lesion_prior_pet + (1.0 - pet_reliability) * lesion_prior_ct
        lesion_logit = torch.logit(lesion_prior.clamp(1e-4, 1.0 - 1e-4))

        boundary_feat = self.ct_boundary_head(feat_ct)
        uncertainty_logit = self.uncertainty_head(torch.cat([base, torch.abs(feat_ct - feat_pet)], dim=1))
        certainty = 1.0 - torch.sigmoid(uncertainty_logit)

        guided_boundary = lesion_prior * boundary_feat
        delta = self.delta_proj(guided_boundary)
        warm_gate = self._warm_gate(base)
        residual_gate = warm_gate * certainty * pet_reliability
        out = base + residual_gate * delta

        aux = {
            f'{prefix}_base': base,
            f'{prefix}_lesion_logit': lesion_logit,
            f'{prefix}_lesion_logit_pet': lesion_logit_pet,
            f'{prefix}_lesion_logit_ct': lesion_logit_ct,
            f'{prefix}_lesion_prior': lesion_prior,
            f'{prefix}_lesion_prior_pet': lesion_prior_pet,
            f'{prefix}_lesion_prior_ct': lesion_prior_ct,
            f'{prefix}_pet_reliability': pet_reliability,
            f'{prefix}_boundary_feat': boundary_feat,
            f'{prefix}_uncertainty_logit': uncertainty_logit,
            f'{prefix}_certainty': certainty,
            f'{prefix}_guided_boundary': guided_boundary,
            f'{prefix}_delta': delta,
            f'{prefix}_gate': residual_gate,
            f'{prefix}_warm_gate': warm_gate.expand(base.size(0), 1, 1, 1),
            f'{prefix}_edge_logit': lesion_logit + certainty,
        }
        return out, aux




class DualSpectrumDisentangler(nn.Module):
    def __init__(self, channels, low_ratio=0.15):
        super().__init__()
        self.low_ratio = low_ratio
        self.low_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def _masks(self, h, w, device):
        fy = torch.linspace(-1.0, 1.0, h, device=device)
        fx = torch.linspace(0.0, 1.0, w // 2 + 1, device=device)
        yy, xx = torch.meshgrid(fy, fx, indexing='ij')
        rr = torch.sqrt(xx.square() + yy.square())
        low = (rr <= self.low_ratio).float()[None, None]
        high = 1.0 - low
        return low, high

    def forward(self, feat):
        _, _, h, w = feat.shape
        fft_feat = torch.fft.rfft2(feat, norm='ortho')
        amp = torch.abs(fft_feat)
        phase = torch.angle(fft_feat)
        low_mask, high_mask = self._masks(h, w, feat.device)

        amp_low = amp * low_mask
        amp_high = amp * high_mask
        real_low = torch.fft.irfft2(torch.polar(amp_low, phase), s=(h, w), norm='ortho')
        real_high = torch.fft.irfft2(torch.polar(amp_high, phase), s=(h, w), norm='ortho')
        recon_full = torch.fft.irfft2(torch.polar(amp_low + amp_high, phase), s=(h, w), norm='ortho')

        return self.low_proj(real_low), self.high_proj(real_high), {
            'real_low': real_low,
            'real_high': real_high,
            'recon_full': recon_full,
        }


class CrossModalReliabilityModulator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 8, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, ct_low, ct_high, pet_low, pet_high):
        gates = self.gate(torch.cat([ct_low, ct_high, pet_low, pet_high], dim=1))
        g_ct_low, g_ct_high, g_pet_low, g_pet_high = torch.chunk(gates, 4, dim=1)
        return {
            'g_ct_low': g_ct_low,
            'g_ct_high': g_ct_high,
            'g_pet_low': g_pet_low,
            'g_pet_high': g_pet_high,
        }


class StructureConstrainedLowFreqFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        bottleneck = max(channels // 2, 64)
        self.structure_head = nn.Sequential(
            nn.Conv2d(channels, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.fuse_reduce = nn.Sequential(
            nn.Conv2d(channels * 3, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
        )
        self.fuse_dw = DepthwiseSeparableConv(bottleneck, channels)
        self.residual = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_norm = DepthwiseSeparableConv(channels, channels)

    def forward(self, ct_low, ct_high, pet_low, pet_high, gates):
        low_fused = gates['g_pet_low'] * pet_low + gates['g_ct_low'] * ct_low
        high_fused = gates['g_ct_high'] * ct_high + gates['g_pet_high'] * pet_high
        structure_support = self.structure_head(ct_high)
        fused = self.fuse_dw(self.fuse_reduce(torch.cat([low_fused, high_fused, structure_support], dim=1)))
        base = self.residual(torch.cat([ct_low + ct_high, pet_low + pet_high], dim=1))
        out = self.out_norm(fused + base)
        return out, {
            'low_fused': low_fused,
            'high_fused': high_fused,
            'structure_support': structure_support,
        }


class MedicalFrequencyFusionBlock(nn.Module):
    def __init__(self, channels, low_ratio=0.15, aux_prefix='x4', enable_edge=True):
        super().__init__()
        self.aux_prefix = aux_prefix
        self.enable_edge = enable_edge
        self.ct_dsd = DualSpectrumDisentangler(channels, low_ratio=low_ratio)
        self.pet_dsd = DualSpectrumDisentangler(channels, low_ratio=low_ratio)
        self.crm = CrossModalReliabilityModulator(channels)
        self.slgf = StructureConstrainedLowFreqFusion(channels)
        self.edge_head = nn.Conv2d(channels, 1, 1) if enable_edge else None

    def forward(self, feat_ct, feat_pet):
        ct_low, ct_high, ct_aux = self.ct_dsd(feat_ct)
        pet_low, pet_high, pet_aux = self.pet_dsd(feat_pet)
        gates = self.crm(ct_low, ct_high, pet_low, pet_high)
        fused, fuse_aux = self.slgf(ct_low, ct_high, pet_low, pet_high, gates)
        prefix = self.aux_prefix
        aux = {
            f'{prefix}_ct_low': ct_low,
            f'{prefix}_ct_high': ct_high,
            f'{prefix}_pet_low': pet_low,
            f'{prefix}_pet_high': pet_high,
            f'{prefix}_ct_recon': ct_aux['recon_full'],
            f'{prefix}_pet_recon': pet_aux['recon_full'],
            f'{prefix}_ct_real_low': ct_aux['real_low'],
            f'{prefix}_ct_real_high': ct_aux['real_high'],
            f'{prefix}_pet_real_low': pet_aux['real_low'],
            f'{prefix}_pet_real_high': pet_aux['real_high'],
            f'{prefix}_g_ct_low': gates['g_ct_low'],
            f'{prefix}_g_ct_high': gates['g_ct_high'],
            f'{prefix}_g_pet_low': gates['g_pet_low'],
            f'{prefix}_g_pet_high': gates['g_pet_high'],
            f'{prefix}_low_fused': fuse_aux['low_fused'],
            f'{prefix}_high_fused': fuse_aux['high_fused'],
            f'{prefix}_structure_support': fuse_aux['structure_support'],
        }
        if self.edge_head is not None:
            aux[f'{prefix}_edge_logit'] = self.edge_head(fuse_aux['structure_support'])
        return fused, aux


class SinglePVTEMCAD(nn.Module):
    """Student: single-stream CT (3-ch), PVTv2-b0 + EMCAD."""
    def __init__(self, pretrained_path=None, in_channels=3, out_channels=1,
                 kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True,
                 lgag_ks=3, activation='relu6'):
        super().__init__()
        self.encoder = timm.create_model(
            'pvt_v2_b0', pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=in_channels
        )
        if pretrained_path:
            load_local_weights_safe(self.encoder, pretrained_path, name='Student_Encoder')

        c1, c2, c3, c4 = self.encoder.feature_info.channels()
        self.decoder = EMCADDecoder(
            channels=(c4, c3, c2, c1),
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            lgag_ks=lgag_ks,
            activation=activation,
        )
        self.out_head4 = nn.Conv2d(c4, out_channels, 1)
        self.out_head3 = nn.Conv2d(c3, out_channels, 1)
        self.out_head2 = nn.Conv2d(c2, out_channels, 1)
        self.out_head1 = nn.Conv2d(c1, out_channels, 1)

    def forward(self, ct, pet=None, target_size=None):
        if ct.shape[1] == 1:
            ct = ct.repeat(1, 3, 1, 1)
        x1, x2, x3, x4 = self.encoder(ct)
        d4, d3, d2, d1 = self.decoder(x4, [x3, x2, x1])
        if target_size is None:
            target_size = ct.shape[-2:]
        p4 = F.interpolate(self.out_head4(d4), size=target_size, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d3), size=target_size, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d2), size=target_size, mode='bilinear', align_corners=False)
        p1 = F.interpolate(self.out_head1(d1), size=target_size, mode='bilinear', align_corners=False)
        return [p1, p2, p3, p4]


class DualPVTB2EMCAD(nn.Module):
    """Teacher: dual-stream CT/PET with baseline add skips and switchable stage-wise enhancements."""
    def __init__(self, pretrained_path=None, in_channels=3, out_channels=1,
                 kernel_sizes=(1, 3, 5), expansion_factor=2, dw_parallel=True, add=True,
                 lgag_ks=3, activation='relu6', enable_freq_v15=True, enable_x4_residual_refine=False,
                 enable_x4_zero_adapter=False, enable_x4_gated_lite=False, enable_freq_x3=False, enable_freq_x2=False,
                 enable_freq_x1=False, alpha_x4_refine=0.25, alpha_freq_x3=0.30,
                 alpha_freq_x2=0.20, alpha_freq_x1=0.15, freq_low_ratio=0.15, x4_lite_warmup_epochs=5, x4_lite_gate_init=-8.0,
                 enable_plbu_lite=False, enable_plbu_x4=False, enable_plbu_x3=False, enable_plbu_x2=False, enable_plbu_x1=False,
                 plbu_warmup_epochs=3, plbu_gate_init=-6.0, plbu_reduction=4,
                 enable_pet_reliability_gate=True, pet_reliability_floor=0.15, pet_reliability_scale=8.0):
        super().__init__()
        self.enc_ct = timm.create_model('pvt_v2_b2', pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=in_channels)
        self.enc_pet = timm.create_model('pvt_v2_b2', pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=in_channels)
        if pretrained_path:
            load_local_weights_safe(self.enc_ct, pretrained_path, name='Teacher_CT_Encoder')
            load_local_weights_safe(self.enc_pet, pretrained_path, name='Teacher_PET_Encoder')

        c1, c2, c3, c4 = self.enc_ct.feature_info.channels()
        self.decoder = EMCADDecoder(
            channels=(c4, c3, c2, c1),
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            lgag_ks=lgag_ks,
            activation=activation,
        )
        self.out_head4 = nn.Conv2d(c4, out_channels, 1)
        self.out_head3 = nn.Conv2d(c3, out_channels, 1)
        self.out_head2 = nn.Conv2d(c2, out_channels, 1)
        self.out_head1 = nn.Conv2d(c1, out_channels, 1)

        if enable_plbu_lite and enable_plbu_x1:
            self.freq_fuse1 = PLBUFusionLite(c1, aux_prefix='x1', warmup_epochs=plbu_warmup_epochs, gate_init=plbu_gate_init, reduction=plbu_reduction, enable_pet_reliability_gate=enable_pet_reliability_gate, pet_reliability_floor=pet_reliability_floor, pet_reliability_scale=pet_reliability_scale)
        else:
            self.freq_fuse1 = WeakFrequencyInjector(c1, low_ratio=freq_low_ratio, aux_prefix='x1', alpha=alpha_freq_x1) if enable_freq_v15 and enable_freq_x1 else None

        if enable_plbu_lite and enable_plbu_x2:
            self.freq_fuse2 = PLBUFusionLite(c2, aux_prefix='x2', warmup_epochs=plbu_warmup_epochs, gate_init=plbu_gate_init, reduction=plbu_reduction, enable_pet_reliability_gate=enable_pet_reliability_gate, pet_reliability_floor=pet_reliability_floor, pet_reliability_scale=pet_reliability_scale)
        else:
            self.freq_fuse2 = WeakFrequencyInjector(c2, low_ratio=freq_low_ratio, aux_prefix='x2', alpha=alpha_freq_x2) if enable_freq_v15 and enable_freq_x2 else None

        if enable_plbu_lite and enable_plbu_x3:
            self.freq_fuse3 = PLBUFusionLite(c3, aux_prefix='x3', warmup_epochs=plbu_warmup_epochs, gate_init=plbu_gate_init, reduction=plbu_reduction, enable_pet_reliability_gate=enable_pet_reliability_gate, pet_reliability_floor=pet_reliability_floor, pet_reliability_scale=pet_reliability_scale)
        else:
            self.freq_fuse3 = WeakFrequencyInjector(c3, low_ratio=freq_low_ratio, aux_prefix='x3', alpha=alpha_freq_x3) if enable_freq_v15 and enable_freq_x3 else None

        if enable_plbu_lite and enable_plbu_x4:
            self.freq_fuse4 = PLBUFusionLite(c4, aux_prefix='x4', warmup_epochs=plbu_warmup_epochs, gate_init=plbu_gate_init, reduction=plbu_reduction, enable_pet_reliability_gate=enable_pet_reliability_gate, pet_reliability_floor=pet_reliability_floor, pet_reliability_scale=pet_reliability_scale)
        elif enable_x4_zero_adapter:
            self.freq_fuse4 = ZeroInitX4Adapter(c4, aux_prefix='x4')
        elif enable_x4_gated_lite:
            self.freq_fuse4 = GatedWarmupX4Lite(c4, low_ratio=freq_low_ratio, aux_prefix='x4', warmup_epochs=x4_lite_warmup_epochs, gate_init=x4_lite_gate_init)
        elif enable_x4_residual_refine:
            self.freq_fuse4 = MedicalFrequencyFusionBlock(c4, low_ratio=freq_low_ratio, aux_prefix='x4', enable_edge=True)
        elif enable_freq_v15:
            self.freq_fuse4 = MedicalFrequencyFusionBlock(c4, low_ratio=freq_low_ratio, aux_prefix='x4', enable_edge=True)
        else:
            self.freq_fuse4 = None

    def forward(self, ct, pet, target_size=None):
        if ct.shape[1] == 1:
            ct = ct.repeat(1, 3, 1, 1)
        if pet.shape[1] == 1:
            pet = pet.repeat(1, 3, 1, 1)
        feats_ct = self.enc_ct(ct)
        feats_pet = self.enc_pet(pet)
        freq_aux = {}
        if self.freq_fuse1 is None:
            x1 = feats_ct[0] + feats_pet[0]
        else:
            x1, aux1 = self.freq_fuse1(feats_ct[0], feats_pet[0])
            freq_aux.update(aux1)
        if self.freq_fuse2 is None:
            x2 = feats_ct[1] + feats_pet[1]
        else:
            x2, aux2 = self.freq_fuse2(feats_ct[1], feats_pet[1])
            freq_aux.update(aux2)
        if self.freq_fuse3 is None:
            x3 = feats_ct[2] + feats_pet[2]
        else:
            x3, aux3 = self.freq_fuse3(feats_ct[2], feats_pet[2])
            freq_aux.update(aux3)
        if self.freq_fuse4 is None:
            x4 = feats_ct[3] + feats_pet[3]
        else:
            x4, freq_aux4 = self.freq_fuse4(feats_ct[3], feats_pet[3])
            freq_aux.update(freq_aux4)
        d4, d3, d2, d1 = self.decoder(x4, [x3, x2, x1])
        if target_size is None:
            target_size = ct.shape[-2:]
        p4 = F.interpolate(self.out_head4(d4), size=target_size, mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.out_head3(d3), size=target_size, mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_head2(d2), size=target_size, mode='bilinear', align_corners=False)
        p1 = F.interpolate(self.out_head1(d1), size=target_size, mode='bilinear', align_corners=False)
        return {'preds': [p1, p2, p3, p4], 'aux': freq_aux}

    def set_epoch(self, epoch):
        for module in (self.freq_fuse1, self.freq_fuse2, self.freq_fuse3, self.freq_fuse4):
            if hasattr(module, 'set_epoch'):
                module.set_epoch(epoch)


def build_mdt_seg_teacher(config):
    p_path = getattr(config, 'pretrained_path', None)
    model = DualPVTB2EMCAD(
        pretrained_path=p_path,
        in_channels=3,
        out_channels=1,
        enable_freq_v15=getattr(config, 'enable_freq_v15', True),
        enable_x4_residual_refine=getattr(config, 'enable_x4_residual_refine', False),
        enable_x4_zero_adapter=getattr(config, 'enable_x4_zero_adapter', False),
        enable_x4_gated_lite=getattr(config, 'enable_x4_gated_lite', False),
        enable_freq_x3=getattr(config, 'enable_freq_x3', False),
        enable_freq_x2=getattr(config, 'enable_freq_x2', False),
        enable_freq_x1=getattr(config, 'enable_freq_x1', False),
        alpha_x4_refine=getattr(config, 'alpha_x4_refine', 0.25),
        alpha_freq_x3=getattr(config, 'alpha_freq_x3', 0.30),
        alpha_freq_x2=getattr(config, 'alpha_freq_x2', 0.20),
        alpha_freq_x1=getattr(config, 'alpha_freq_x1', 0.15),
        freq_low_ratio=getattr(config, 'freq_low_ratio', 0.15),
        x4_lite_warmup_epochs=getattr(config, 'x4_lite_warmup_epochs', 5),
        x4_lite_gate_init=getattr(config, 'x4_lite_gate_init', -8.0),
        enable_plbu_lite=getattr(config, 'enable_plbu_lite', False),
        enable_plbu_x4=getattr(config, 'enable_plbu_x4', False),
        enable_plbu_x3=getattr(config, 'enable_plbu_x3', False),
        enable_plbu_x2=getattr(config, 'enable_plbu_x2', False),
        enable_plbu_x1=getattr(config, 'enable_plbu_x1', False),
        plbu_warmup_epochs=getattr(config, 'plbu_warmup_epochs', 3),
        plbu_gate_init=getattr(config, 'plbu_gate_init', -6.0),
        plbu_reduction=getattr(config, 'plbu_reduction', 4),
        enable_pet_reliability_gate=getattr(config, 'enable_pet_reliability_gate', True),
        pet_reliability_floor=getattr(config, 'pet_reliability_floor', 0.15),
        pet_reliability_scale=getattr(config, 'pet_reliability_scale', 8.0),
    )
    return dict(model=model)


def build_mdt_seg_student(config):
    s_path = getattr(config, 'student_pretrained_path', None) or getattr(config, 'pretrained_path', None)
    model = SinglePVTEMCAD(pretrained_path=s_path, in_channels=3, out_channels=1)
    return dict(model=model)
