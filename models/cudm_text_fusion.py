# -*- coding: utf-8 -*-
"""AlignedCUDMTextFusion v2: explicit disentanglement + BCG-PA text fusion.

Selectable module modes:
    extractor_only: Module I only, no text cross-attention fusion.
    text_only:     Module II only, using aligned CT/PET as pseudo specific branches.
    dual:          Module I + Module II, full design.
"""
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_local_text_feature(text_tower_path: str, prompt: str, hidden_size: int = 768) -> torch.Tensor:
    tower_dir = Path(text_tower_path).expanduser().resolve()
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(str(tower_dir), local_files_only=True)
        bert = AutoModel.from_pretrained(str(tower_dir), local_files_only=True)
        bert.eval()
        enc = tokenizer([prompt], padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            out = bert(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            feat = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            feat = F.normalize(feat, dim=-1)
        print(f'[+] Aligned-CUDM-Text v2: text feature shape {tuple(feat.shape)}')
        return feat.cpu()
    except Exception as exc:
        print(f'[!] Aligned-CUDM-Text v2: failed to load text tower ({exc}), using random init.')
        rand = torch.randn(1, hidden_size)
        return F.normalize(rand, dim=-1)


class ConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, stride=1, groups=1, dilation=1):
        super().__init__()
        padding = (k // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding,
                      groups=groups, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ExplicitDisentangledTumorExtractor(nn.Module):
    """Module I: explicit Common / CT-specific / PET-specific disentanglement."""

    def __init__(self, channels: int):
        super().__init__()
        c = channels
        mid = max(1, c // 2)
        squeeze = max(1, c // 4)

        self.common_encoder = nn.Sequential(
            ConvBN(c * 2, c, k=5),
            ConvBN(c, c, k=3),
            ConvBN(c, c, k=1),
        )
        self.common_channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, squeeze, 1), nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, c, 1), nn.Sigmoid(),
        )

        self.ct_diff_encoder = nn.Sequential(
            ConvBN(c * 2, c, k=1),
            ConvBN(c, mid, k=3, dilation=2),
            nn.Conv2d(mid, 1, 1), nn.Sigmoid(),
        )
        self.ct_specific_transform = nn.Sequential(ConvBN(c, c, k=3), ConvBN(c, c, k=1))

        self.pet_diff_encoder = nn.Sequential(
            ConvBN(c * 2, c, k=1),
            ConvBN(c, mid, k=3, dilation=2),
            nn.Conv2d(mid, 1, 1), nn.Sigmoid(),
        )
        self.pet_specific_transform = nn.Sequential(ConvBN(c, c, k=3), ConvBN(c, c, k=1))

    def forward(self, ct, pet):
        b, c, h, w = ct.shape
        concat = torch.cat([ct, pet], dim=1)

        common_raw = self.common_encoder(concat)
        ch_gate = self.common_channel_gate(common_raw)
        common = common_raw * ch_gate

        ct_saliency = self.ct_diff_encoder(concat)
        pet_saliency = self.pet_diff_encoder(concat)
        ct_specific = self.ct_specific_transform(ct) * ct_saliency
        pet_specific = self.pet_specific_transform(pet) * pet_saliency

        common_flat = common.reshape(b, c, -1)
        ct_flat = ct_specific.reshape(b, c, -1)
        pet_flat = pet_specific.reshape(b, c, -1)
        common_energy = common_flat.pow(2).sum(dim=2, keepdim=True).clamp_min(1e-6)
        ct_proj = (ct_flat * common_flat).sum(dim=2, keepdim=True) / common_energy
        pet_proj = (pet_flat * common_flat).sum(dim=2, keepdim=True) / common_energy
        ct_specific = (ct_flat - ct_proj * common_flat).reshape(b, c, h, w)
        pet_specific = (pet_flat - pet_proj * common_flat).reshape(b, c, h, w)

        aux = {
            'common_channel_gate': ch_gate.detach(),
            'ct_diff_saliency': ct_saliency.detach(),
            'pet_diff_saliency': pet_saliency.detach(),
            'ortho_residual': (ct_proj.abs().mean() + pet_proj.abs().mean()).detach(),
        }
        return common, ct_specific, pet_specific, aux


class BCGPATextGuidedFusion(nn.Module):
    """Module II: bidirectional cross-attention, FiLM and semantic 3-way gates."""

    def __init__(self, channels: int, text_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        c = channels
        num_heads = max(1, math.gcd(c, int(num_heads)))
        self.img_proj = nn.Conv2d(c, c, 1)
        self.text_proj_q = nn.Linear(text_dim, c)
        self.text_proj_kv = nn.Linear(text_dim, c)
        self.text_to_img_attn = nn.MultiheadAttention(c, num_heads, dropout=dropout, batch_first=True)
        self.img_to_text_attn = nn.MultiheadAttention(c, num_heads, dropout=dropout, batch_first=True)
        self.film = nn.Sequential(nn.Linear(text_dim, c * 2), nn.LayerNorm(c * 2))
        self.gate = nn.Sequential(ConvBN(c * 4, c, k=1), nn.Conv2d(c, 3, 1), nn.Softmax(dim=1))
        self.out_proj = nn.Sequential(ConvBN(c, c, k=1), nn.Dropout2d(dropout))

    def forward(self, common, ct_specific, pet_specific, text_feat):
        b, c, h, w = common.shape
        text_feat = text_feat.to(dtype=common.dtype, device=common.device)
        if text_feat.size(0) == 1 and b > 1:
            text_feat = text_feat.expand(b, -1)

        img_seq = self.img_proj(common).reshape(b, c, h * w).permute(0, 2, 1)
        text_q = self.text_proj_q(text_feat).unsqueeze(1)
        text_kv = self.text_proj_kv(text_feat).unsqueeze(1)

        _, text_to_img_w = self.text_to_img_attn(
            text_q, img_seq, img_seq, need_weights=True, average_attn_weights=True
        )
        pixel_enhanced, _ = self.img_to_text_attn(img_seq, text_kv, text_kv)
        pixel_enhanced = pixel_enhanced.permute(0, 2, 1).reshape(b, c, h, w)
        text_attn_map = text_to_img_w.reshape(b, 1, h, w)

        gamma, beta = self.film(text_feat).chunk(2, dim=-1)
        gamma = gamma.reshape(b, c, 1, 1)
        beta = beta.reshape(b, c, 1, 1)
        common_modulated = common * (1 + torch.tanh(gamma)) + torch.tanh(beta)

        gates = self.gate(torch.cat([common_modulated, ct_specific, pet_specific, pixel_enhanced], dim=1))
        fused = gates[:, 0:1] * common_modulated + gates[:, 1:2] * ct_specific + gates[:, 2:3] * pet_specific
        out = self.out_proj(fused)
        aux = {
            'text_pixel_attn_map': text_attn_map.detach(),
            'fusion_gates': gates.detach(),
            'film_gamma': gamma.detach(),
            'film_beta': beta.detach(),
        }
        return out, aux


class AlignedCUDMTextBlock(nn.Module):
    def __init__(self, ct_channels, pet_channels, out_channels, text_feat,
                 num_heads=8, dropout=0.1, module_mode='dual'):
        super().__init__()
        if module_mode not in ('extractor_only', 'text_only', 'dual'):
            raise ValueError(f'Unsupported module_mode={module_mode}')
        text_dim = int(text_feat.shape[-1])
        self.module_mode = module_mode
        self.ct_proj = nn.Sequential(
            nn.Conv2d(ct_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.pet_proj = nn.Sequential(
            nn.Conv2d(pet_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.extractor = ExplicitDisentangledTumorExtractor(out_channels) if module_mode in ('extractor_only', 'dual') else None
        self.fusion_block = BCGPATextGuidedFusion(out_channels, text_dim, num_heads=num_heads, dropout=dropout) if module_mode in ('text_only', 'dual') else None
        self.register_buffer('text_embed', text_feat.float())
        self.res_scale = nn.Parameter(torch.zeros(1))
        self.cache_visuals = False
        self._last_visuals = {}

    def forward(self, ct_feat, pet_feat):
        if self.cache_visuals:
            self._last_visuals = {'ct_encoder': ct_feat.detach().cpu(), 'pet_encoder': pet_feat.detach().cpu()}

        ct = self.ct_proj(ct_feat)
        pet = self.pet_proj(pet_feat)
        if pet.shape[-2:] != ct.shape[-2:]:
            pet = F.interpolate(pet, size=ct.shape[-2:], mode='bilinear', align_corners=False)
        base = (ct + pet) * 0.5

        aux = {}
        if self.module_mode == 'text_only':
            common, ct_tumor, pet_tumor = base, ct, pet
        else:
            common, ct_tumor, pet_tumor, extract_aux = self.extractor(ct, pet)
            aux.update(extract_aux)

        if self.module_mode == 'extractor_only':
            fused_delta = common + 0.5 * (ct_tumor + pet_tumor)
        else:
            fused_delta, fusion_aux = self.fusion_block(common, ct_tumor, pet_tumor, self.text_embed)
            aux.update(fusion_aux)

        fused = base + torch.tanh(self.res_scale) * (fused_delta - base)
        if self.cache_visuals:
            self._last_visuals.update({
                'ct_aligned': ct.detach().cpu(),
                'pet_aligned': pet.detach().cpu(),
                'common': common.detach().cpu(),
                'ct_tumor': ct_tumor.detach().cpu(),
                'pet_tumor': pet_tumor.detach().cpu(),
                'fused': fused.detach().cpu(),
            })
            for key in ('ct_diff_saliency', 'pet_diff_saliency', 'text_pixel_attn_map', 'fusion_gates'):
                if key in aux and isinstance(aux[key], torch.Tensor):
                    self._last_visuals[key] = aux[key].detach().cpu()
        return fused, aux


class AlignedCUDMTextFusion(nn.Module):
    def __init__(self, ct_channels, pet_channels, out_channels,
                 prompt='focal abnormal metabolic lung lesion on PET-CT scan',
                 text_tower_path='/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower',
                 num_heads=8, dropout=0.1, module_mode='dual'):
        super().__init__()
        if not (len(ct_channels) == len(pet_channels) == len(out_channels)):
            raise ValueError('ct_channels, pet_channels and out_channels must have equal length.')
        text_feat = _load_local_text_feature(text_tower_path, prompt)
        self.blocks = nn.ModuleList([
            AlignedCUDMTextBlock(ct_ch, pet_ch, out_ch, text_feat,
                                 num_heads=num_heads, dropout=dropout, module_mode=module_mode)
            for ct_ch, pet_ch, out_ch in zip(ct_channels, pet_channels, out_channels)
        ])
        self.cache_visuals = False
        self.module_mode = module_mode

    def set_visuals(self, enabled: bool):
        self.cache_visuals = bool(enabled)
        for blk in self.blocks:
            blk.cache_visuals = self.cache_visuals
            if self.cache_visuals:
                blk._last_visuals = {}

    def get_fusion_visuals(self):
        visuals = {}
        for idx, blk in enumerate(self.blocks, start=1):
            if blk._last_visuals:
                visuals[f'cudm_s{idx}'] = dict(blk._last_visuals)
        return visuals

    def forward(self, ct_feats, pet_feats):
        fused_list, aux_list = [], []
        for blk, ct_f, pet_f in zip(self.blocks, ct_feats, pet_feats):
            fused, aux = blk(ct_f, pet_f)
            fused_list.append(fused)
            aux_list.append(aux)
        return fused_list, aux_list
