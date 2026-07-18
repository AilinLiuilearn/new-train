from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

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
    """Bidirectional Cross-scale Orthogonal Residual Transport."""

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
        self.gamma = nn.ParameterList([
            nn.Parameter(torch.zeros(1, c, 1, 1)) for c in self.stage_channels
        ])

    def _check_inputs(self, feats: Sequence[torch.Tensor]) -> None:
        if len(feats) != 4:
            raise ValueError(f"BCORT expects 4 input tensors, got {len(feats)}")
        for i, (feat, ch) in enumerate(zip(feats, self.stage_channels), start=1):
            if not torch.is_tensor(feat):
                raise TypeError(f"Stage S{i} is not a tensor")
            if feat.ndim != 4:
                raise ValueError(f"Stage S{i} must be NCHW, got shape {tuple(feat.shape)}")
            if feat.shape[1] != ch:
                raise ValueError(f"Stage S{i} channel mismatch: expected {ch}, got {feat.shape[1]}")

    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(x.float() ** 2) + 1e-12)

    def _orth_residual(self, candidate: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        numerator = torch.sum(candidate * ref, dim=1, keepdim=True)
        denominator = torch.sum(ref * ref, dim=1, keepdim=True) + self.eps
        parallel = numerator / denominator * ref
        return candidate - parallel

    def _match_rms(self, residual: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        r_rms = self._rms(residual)
        ref_rms = self._rms(ref)
        scale = ref_rms / (r_rms + self.eps)
        return residual * scale

    def _diag_item(self, gamma: torch.Tensor, td: torch.Tensor, bu: torch.Tensor, transport: torch.Tensor, injection: torch.Tensor, ref: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "gamma_abs_mean": gamma.abs().mean(),
            "td_rms": self._rms(td),
            "bu_rms": self._rms(bu),
            "transport_rms": self._rms(transport),
            "injection_rms": self._rms(injection),
            "injection_feature_ratio": self._rms(injection) / (self._rms(ref) + self.eps),
        }

    def forward(self, feats: Sequence[torch.Tensor], return_diagnostics: bool = False):
        self._check_inputs(feats)
        s1, s2, s3, s4 = feats

        td3_cand = self.td_align[0](s4, s3.shape[-2:])
        td3 = self._match_rms(self._orth_residual(td3_cand, s3), s3)
        td2_cand = self.td_align[1](td3, s2.shape[-2:])
        td2 = self._match_rms(self._orth_residual(td2_cand, s2), s2)
        td1_cand = self.td_align[2](td2, s1.shape[-2:])
        td1 = self._match_rms(self._orth_residual(td1_cand, s1), s1)

        bu2_cand = self.bu_align[0](s1, s2.shape[-2:])
        bu2 = self._match_rms(self._orth_residual(s2 - bu2_cand, s2), s2)
        bu3_cand = self.bu_align[1](bu2, s3.shape[-2:])
        bu3 = self._match_rms(self._orth_residual(s3 - bu3_cand, s3), s3)
        bu4_cand = self.bu_align[2](bu3, s4.shape[-2:])
        bu4 = self._match_rms(self._orth_residual(s4 - bu4_cand, s4), s4)

        transport1 = td1
        transport2 = 0.5 * (td2 + bu2)
        transport3 = 0.5 * (td3 + bu3)
        transport4 = bu4
        injections = [self.gamma[i] * t for i, t in enumerate([transport1, transport2, transport3, transport4])]

        outs = [s1 + injections[0], s2 + injections[1], s3 + injections[2], s4 + injections[3]]

        for i, out in enumerate(outs, start=1):
            if not torch.isfinite(out).all():
                raise RuntimeError(f"BCORT output stage S{i} contains non-finite values")

        diagnostics = {
            "s1_gamma_abs_mean": self.gamma[0].abs().mean(),
            "s1_td_rms": self._rms(td1),
            "s1_bu_rms": self._rms(torch.zeros_like(s1)),
            "s1_transport_rms": self._rms(transport1),
            "s1_injection_rms": self._rms(injections[0]),
            "s1_injection_feature_ratio": self._rms(injections[0]) / (self._rms(s1) + self.eps),
            "s2_gamma_abs_mean": self.gamma[1].abs().mean(),
            "s2_td_rms": self._rms(td2),
            "s2_bu_rms": self._rms(bu2),
            "s2_transport_rms": self._rms(transport2),
            "s2_injection_rms": self._rms(injections[1]),
            "s2_injection_feature_ratio": self._rms(injections[1]) / (self._rms(s2) + self.eps),
            "s3_gamma_abs_mean": self.gamma[2].abs().mean(),
            "s3_td_rms": self._rms(td3),
            "s3_bu_rms": self._rms(bu3),
            "s3_transport_rms": self._rms(transport3),
            "s3_injection_rms": self._rms(injections[2]),
            "s3_injection_feature_ratio": self._rms(injections[2]) / (self._rms(s3) + self.eps),
            "s4_gamma_abs_mean": self.gamma[3].abs().mean(),
            "s4_td_rms": self._rms(s4),
            "s4_bu_rms": self._rms(bu4),
            "s4_transport_rms": self._rms(transport4),
            "s4_injection_rms": self._rms(injections[3]),
            "s4_injection_feature_ratio": self._rms(injections[3]) / (self._rms(s4) + self.eps),
        }

        if return_diagnostics:
            return outs, diagnostics
        return outs
