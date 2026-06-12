# -*- coding: utf-8 -*-
"""Ablation version of MPA-BioCLIP without text participation.

This module keeps the same visual cross-modal alignment path as `mpa_bioclip.py`
but removes text prompt injection entirely so the effect of text guidance can be
isolated in ablation experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPABioCLIPNoTextBlock(nn.Module):
    def __init__(self, ct_channels, pet_channels, out_channels, mlp_ratio=0.25, dropout=0.1):
        super().__init__()
        hidden = max(1, int(out_channels * mlp_ratio))
        gate_hidden = max(1, out_channels // 4)

        self.ct_proj = nn.Sequential(
            nn.Conv2d(ct_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pet_proj = nn.Sequential(
            nn.Conv2d(pet_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.pet_guide_ct = nn.Sequential(
            nn.Conv2d(out_channels, gate_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.ct_guide_pet = nn.Sequential(
            nn.Conv2d(out_channels, gate_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.ct_down = nn.Linear(out_channels, hidden)
        self.pet_down = nn.Linear(out_channels, hidden)
        self.up = nn.Linear(hidden, out_channels)
        self.act = nn.GELU()

        self.gamma_ct = nn.Parameter(torch.ones(1))
        self.beta_ct = nn.Parameter(torch.zeros(1))
        self.gamma_pet = nn.Parameter(torch.ones(1))
        self.beta_pet = nn.Parameter(torch.zeros(1))
        self.s = nn.Parameter(torch.ones(1))

        self.adapter_ct = nn.Sequential(
            nn.Linear(out_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels),
        )
        self.adapter_pet = nn.Sequential(
            nn.Linear(out_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels),
        )
        self.cache_visuals = False
        self._last_visuals = {}

    def forward(self, ct_feat, pet_feat):
        if self.cache_visuals:
            self._last_visuals = {
                'ct_encoder': ct_feat.detach().cpu(),
                'pet_encoder': pet_feat.detach().cpu(),
            }

        ct = self.ct_proj(ct_feat)
        pet = self.pet_proj(pet_feat)
        if pet.shape[-2:] != ct.shape[-2:]:
            pet = F.interpolate(pet, size=ct.shape[-2:], mode='bilinear', align_corners=False)

        pet_to_ct_gate = self.pet_guide_ct(pet)
        ct_to_pet_gate = self.ct_guide_pet(ct)
        ct_guided = ct * pet_to_ct_gate
        pet_guided = pet * ct_to_pet_gate

        b, c, h, w = ct_guided.shape
        ct_t = ct_guided.flatten(2).transpose(1, 2)
        pet_t = pet_guided.flatten(2).transpose(1, 2)

        p_vis = self.act(self.up(self.ct_down(ct_t) + self.pet_down(pet_t)))
        p_final = p_vis

        p_ct = self.gamma_ct * p_final + self.beta_ct + self.s * p_final
        p_pet = self.gamma_pet * p_final + self.beta_pet + self.s * p_final

        ct_p = ct_t + p_ct
        pet_p = pet_t + p_pet
        ct_out = ct_p + self.adapter_ct(ct_p)
        pet_out = pet_p + self.adapter_pet(pet_p)

        ct_out = ct_out.transpose(1, 2).reshape(b, c, h, w)
        pet_out = pet_out.transpose(1, 2).reshape(b, c, h, w)
        if self.cache_visuals:
            self._last_visuals.update({
                'ct_projected': ct.detach().cpu(),
                'pet_projected': pet.detach().cpu(),
                'pet_to_ct_gate': pet_to_ct_gate.detach().cpu(),
                'ct_to_pet_gate': ct_to_pet_gate.detach().cpu(),
                'ct_no_text_modulated': ct_out.detach().cpu(),
                'pet_no_text_modulated': pet_out.detach().cpu(),
            })
        return ct_out, pet_out


class MPABioCLIPNoTextSumFusion(nn.Module):
    def __init__(self, ct_channels, pet_channels, out_channels, mlp_ratio=0.25):
        super().__init__()
        if not (len(ct_channels) == len(pet_channels) == len(out_channels)):
            raise ValueError('ct_channels, pet_channels and out_channels must have the same length.')
        self.blocks = nn.ModuleList([
            MPABioCLIPNoTextBlock(ct_ch, pet_ch, out_ch, mlp_ratio=mlp_ratio)
            for ct_ch, pet_ch, out_ch in zip(ct_channels, pet_channels, out_channels)
        ])
        self.cache_visuals = False

    def set_visuals(self, enabled):
        self.cache_visuals = bool(enabled)
        for block in self.blocks:
            block.cache_visuals = self.cache_visuals
            if self.cache_visuals:
                block._last_visuals = {}

    def forward(self, ct_feats, pet_feats):
        fused = []
        for block, ct_feat, pet_feat in zip(self.blocks, ct_feats, pet_feats):
            ct_aligned, pet_aligned = block(ct_feat, pet_feat)
            fused.append(ct_aligned + pet_aligned)
        return fused

    def get_fusion_visuals(self):
        visuals = {}
        for idx, block in enumerate(self.blocks, start=1):
            if block._last_visuals:
                visuals[f'mpa_no_text_s{idx}'] = dict(block._last_visuals)
        return visuals
