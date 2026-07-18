from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ChannelMapper(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.mode = str(mode)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] != size:
            if self.mode == "bilinear":
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            elif self.mode == "area":
                x = F.interpolate(x, size=size, mode="area")
            else:
                raise ValueError(f"Unsupported interpolation mode: {self.mode}")
        return self.proj(x)


class BCORT(nn.Module):
    def __init__(self, stage_channels: Sequence[int], eps: float = 1e-6) -> None:
        super().__init__()
        if len(stage_channels) != 4:
            raise ValueError(f"BCORT expects 4 stages, got {len(stage_channels)}")
        self.stage_channels = tuple(int(c) for c in stage_channels)
        self.eps = float(eps)
        self.td_align = nn.ModuleList([
            _ChannelMapper(self.stage_channels[3], self.stage_channels[2], mode="bilinear"),
            _ChannelMapper(self.stage_channels[2], self.stage_channels[1], mode="bilinear"),
            _ChannelMapper(self.stage_channels[1], self.stage_channels[0], mode="bilinear"),
        ])
        self.bu_align = nn.ModuleList([
            _ChannelMapper(self.stage_channels[0], self.stage_channels[1], mode="area"),
            _ChannelMapper(self.stage_channels[1], self.stage_channels[2], mode="area"),
            _ChannelMapper(self.stage_channels[2], self.stage_channels[3], mode="area"),
        ])
        self.gamma = nn.ParameterList([nn.Parameter(torch.zeros(1, c, 1, 1)) for c in self.stage_channels])

    def _check_inputs(self, feats: Sequence[torch.Tensor]) -> None:
        if len(feats) != 4:
            raise ValueError(f"BCORT expects 4 input tensors, got {len(feats)}")
        for i, (feat, ch) in enumerate(zip(feats, self.stage_channels), start=1):
            if feat.ndim != 4 or feat.shape[1] != ch:
                raise ValueError(f"Stage S{i} mismatch: expected NCHW with C={ch}, got {tuple(feat.shape)}")

    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x.float() ** 2) + 1e-12)

    def _orth_residual(self, candidate: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        candidate_fp32 = candidate.float()
        ref_fp32 = ref.float()
        numerator = (candidate_fp32 * ref_fp32).sum(dim=1, keepdim=True)
        denominator = ref_fp32.square().sum(dim=1, keepdim=True) + self.eps
        parallel = numerator / denominator * ref_fp32
        return (candidate_fp32 - parallel).to(candidate.dtype)

    def _cap_local_energy(self, residual: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        residual_fp32 = residual.float()
        ref_fp32 = ref.float()
        residual_rms = torch.sqrt(residual_fp32.square().mean(dim=1, keepdim=True) + self.eps)
        ref_rms = torch.sqrt(ref_fp32.square().mean(dim=1, keepdim=True) + self.eps)
        scale = ref_rms / torch.maximum(residual_rms, ref_rms)
        return (residual_fp32 * scale).to(residual.dtype)

    def forward(self, feats: Sequence[torch.Tensor], return_diagnostics: bool = False):
        self._check_inputs(feats)
        s1, s2, s3, s4 = feats

        top_down = [torch.zeros_like(x) for x in feats]
        semantic_carrier = s4
        for stage_index, mapper_index in ((2, 0), (1, 1), (0, 2)):
            candidate = self.td_align[mapper_index](semantic_carrier, feats[stage_index].shape[-2:])
            complement = self._cap_local_energy(self._orth_residual(candidate, feats[stage_index]), feats[stage_index])
            top_down[stage_index] = complement
            semantic_carrier = feats[stage_index] + complement

        bottom_up = [torch.zeros_like(x) for x in feats]
        coarse_s1 = self.td_align[2](s2, s1.shape[-2:])
        detail_carrier = s1 - coarse_s1
        for stage_index, mapper_index in ((1, 0), (2, 1), (3, 2)):
            candidate = self.bu_align[mapper_index](detail_carrier, feats[stage_index].shape[-2:])
            complement = self._cap_local_energy(self._orth_residual(candidate, feats[stage_index]), feats[stage_index])
            bottom_up[stage_index] = complement
            if stage_index < 3:
                deeper_mapper_index = 2 - stage_index
                coarse_current = self.td_align[deeper_mapper_index](feats[stage_index + 1], feats[stage_index].shape[-2:])
                intrinsic_detail = feats[stage_index] - coarse_current
                detail_carrier = intrinsic_detail + complement

        transport = [
            top_down[0],
            0.5 * (top_down[1] + bottom_up[1]),
            0.5 * (top_down[2] + bottom_up[2]),
            bottom_up[3],
        ]
        injections = [self.gamma[i] * transport[i] for i in range(4)]
        outs = [feats[i] + injections[i] for i in range(4)]
        for i, out in enumerate(outs, start=1):
            if not torch.isfinite(out).all():
                raise RuntimeError(f"BCORT output stage S{i} contains non-finite values")

        diagnostics = {}
        for i in range(4):
            diagnostics.update({
                f"s{i+1}_gamma_abs_mean": self.gamma[i].abs().mean(),
                f"s{i+1}_td_rms": self._rms(top_down[i]),
                f"s{i+1}_bu_rms": self._rms(bottom_up[i]),
                f"s{i+1}_transport_rms": self._rms(transport[i]),
                f"s{i+1}_injection_rms": self._rms(injections[i]),
                f"s{i+1}_injection_feature_ratio": self._rms(injections[i]) / (self._rms(feats[i]) + self.eps),
            })
        return (outs, diagnostics) if return_diagnostics else outs
