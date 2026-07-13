# -*- coding: utf-8 -*-
"""PG-MTR retrieval-only modules for dual-decoder PET-grounded segmentation."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "StagePETGroundedMetabolicTokenRetrieval",
    "PETGroundedMetabolicTokenRetrieval",
]


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    channels = int(channels)
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")
    for groups in range(min(int(max_groups), channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    try:
        return nn.init.trunc_normal_(tensor, mean=0.0, std=float(std), a=-2.0, b=2.0)
    except Exception:
        with torch.no_grad():
            return tensor.normal_(mean=0.0, std=float(std))


def _rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x.float().pow(2).mean().add(eps).sqrt()


def _normalized_assignment_entropy(assignment: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = assignment.float().clamp_min(eps)
    entropy = -(probs * probs.log()).sum(dim=1).mean()
    max_entropy = math.log(float(assignment.shape[1]))
    return entropy / max(max_entropy, eps)


def _assignment_peak(assignment: torch.Tensor) -> torch.Tensor:
    return assignment.float().max(dim=1).values.mean()


def _mean_off_diagonal_token_cosine(tokens: torch.Tensor) -> torch.Tensor:
    num_tokens = int(tokens.shape[0])
    if num_tokens <= 1:
        return tokens.new_zeros((), dtype=torch.float32)
    norm_tokens = F.normalize(tokens.float(), dim=-1, eps=1e-6)
    similarity = norm_tokens @ norm_tokens.transpose(0, 1)
    off_diagonal_sum = similarity.sum() - torch.diagonal(similarity).sum()
    denominator = float(num_tokens * (num_tokens - 1))
    return off_diagonal_sum / denominator


def _balanced_fg_bg_loss(loss_map: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = mask.float()
    if mask.ndim == loss_map.ndim - 1:
        mask = mask.unsqueeze(1)
    fg = mask > 0.5
    bg = ~fg
    eps = 1e-6
    fg_valid = fg.flatten(1).any(dim=1)
    bg_valid = bg.flatten(1).any(dim=1)

    flat_loss = loss_map.flatten(1)
    flat_fg = fg.flatten(1)
    flat_bg = bg.flatten(1)

    fg_mean = torch.zeros(loss_map.shape[0], device=loss_map.device, dtype=loss_map.dtype)
    bg_mean = torch.zeros_like(fg_mean)
    if fg.any():
        fg_mean = (flat_loss * flat_fg.float()).sum(dim=1) / flat_fg.float().sum(dim=1).clamp_min(eps)
    if bg.any():
        bg_mean = (flat_loss * flat_bg.float()).sum(dim=1) / flat_bg.float().sum(dim=1).clamp_min(eps)

    fg_count = fg_valid.float().sum().clamp_min(1.0)
    bg_count = bg_valid.float().sum().clamp_min(1.0)
    fg_loss = (fg_mean * fg_valid.float()).sum() / fg_count
    bg_loss = (bg_mean * bg_valid.float()).sum() / bg_count
    has_fg = fg_valid.any()
    balanced = 0.5 * (fg_loss + bg_loss) if bool(has_fg) else bg_loss
    valid_ratio = fg.float().mean()
    return balanced, fg_loss, bg_loss, valid_ratio


class StagePETGroundedMetabolicTokenRetrieval(nn.Module):
    def __init__(self, in_channels: int, num_tokens: int = 8, latent_dim: Optional[int] = None, temperature: float = 0.07) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_tokens = int(num_tokens)
        self.temperature = float(temperature)
        if self.in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if self.num_tokens <= 1:
            raise ValueError(f"num_tokens must be > 1, got {num_tokens}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if latent_dim is None:
            latent_dim = min(max(self.in_channels // 4, 32), 128)
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")

        self.ct_query_proj = nn.Sequential(
            nn.GroupNorm(_valid_group_count(self.in_channels), self.in_channels),
            nn.Conv2d(self.in_channels, self.latent_dim, kernel_size=1, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for submodule in self.ct_query_proj.modules():
            if isinstance(submodule, nn.Conv2d):
                nn.init.kaiming_normal_(submodule.weight, mode="fan_out", nonlinearity="linear")
                if submodule.bias is not None:
                    nn.init.zeros_(submodule.bias)

    def _validate_ct(self, ct_feat: torch.Tensor) -> None:
        if ct_feat.ndim != 4:
            raise ValueError(f"ct_feat must have shape [B,C,H,W], got {tuple(ct_feat.shape)}")
        if ct_feat.shape[1] != self.in_channels:
            raise ValueError(f"Expected CT channels={self.in_channels}, got {ct_feat.shape[1]}")

    def forward(self, ct_feat: torch.Tensor) -> torch.Tensor:
        self._validate_ct(ct_feat)
        return self.ct_query_proj(ct_feat)


class PETGroundedMetabolicTokenRetrieval(nn.Module):
    def __init__(self, channels_list: Sequence[int], num_tokens: int = 8, temperature: float = 0.07, stage_mode: str = "all") -> None:
        super().__init__()
        channels = [int(c) for c in channels_list]
        if len(channels) != 4:
            raise ValueError(f"PG-MTR expects exactly four encoder stages, got {len(channels)}")
        self.channels_list = channels
        self.num_tokens = int(num_tokens)
        self.temperature = float(temperature)
        self.stage_mode = str(stage_mode)
        if self.stage_mode == "s4":
            active_stage_numbers = (4,)
        elif self.stage_mode in ("s34", "deep"):
            active_stage_numbers = (3, 4)
        elif self.stage_mode == "s234":
            active_stage_numbers = (2, 3, 4)
        elif self.stage_mode == "all":
            active_stage_numbers = (1, 2, 3, 4)
        else:
            raise ValueError(f"Unsupported stage_mode={stage_mode!r}; expected one of ('s4', 's34', 'deep', 's234', 'all')")
        self.active_stage_numbers = active_stage_numbers
        self.active_stage_indices = tuple(stage_number - 1 for stage_number in active_stage_numbers)
        self.writer_stage = max(self.active_stage_numbers)
        writer_index = self.writer_stage - 1
        self.latent_dim = min(max(channels[writer_index] // 4, 32), 128)

        self.stage_modules = nn.ModuleDict()
        for stage_number, stage_index in zip(self.active_stage_numbers, self.active_stage_indices):
            self.stage_modules[str(stage_number)] = StagePETGroundedMetabolicTokenRetrieval(
                in_channels=channels[stage_index],
                num_tokens=self.num_tokens,
                latent_dim=self.latent_dim,
                temperature=self.temperature,
            )

        self.shared_memory_tokens = nn.Parameter(torch.empty(self.num_tokens, self.latent_dim))
        self.shared_token_key = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.shared_token_value = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.pet_query_proj_writer = nn.Sequential(
            nn.GroupNorm(_valid_group_count(channels[writer_index]), channels[writer_index]),
            nn.Conv2d(channels[writer_index], self.latent_dim, kernel_size=1, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _trunc_normal_(self.shared_memory_tokens, std=0.02)
        with torch.no_grad():
            self.shared_memory_tokens.sub_(self.shared_memory_tokens.mean(dim=0, keepdim=True))
        nn.init.xavier_uniform_(self.shared_token_key.weight)
        nn.init.xavier_uniform_(self.shared_token_value.weight)
        for module in (self.pet_query_proj_writer,):
            for submodule in module.modules():
                if isinstance(submodule, nn.Conv2d):
                    nn.init.kaiming_normal_(submodule.weight, mode="fan_out", nonlinearity="linear")
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    @staticmethod
    def _validate_feature_list(features: Sequence[torch.Tensor], name: str) -> None:
        if len(features) != 4:
            raise ValueError(f"{name} must contain four stage features, got {len(features)}")

    @staticmethod
    def _prefix_diagnostics(stage_number: int, diagnostics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {f"pg_mtr_s{stage_number}_{key}": value for key, value in diagnostics.items()}

    def _token_key_value(self, detach: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        token_key = self.shared_token_key(self.shared_memory_tokens)
        token_value = self.shared_token_value(self.shared_memory_tokens)
        if detach:
            token_key = token_key.detach()
            token_value = token_value.detach()
        return token_key, token_value

    def _assignment(self, query: torch.Tensor, token_key: torch.Tensor) -> torch.Tensor:
        query_fp32 = F.normalize(query.float(), dim=1, eps=1e-6)
        key_fp32 = F.normalize(token_key.float(), dim=-1, eps=1e-6)
        logits = torch.einsum("bdhw,kd->bkhw", query_fp32, key_fp32) / self.temperature
        return torch.softmax(logits, dim=1)

    @staticmethod
    def _read_memory(assignment: torch.Tensor, token_value: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        memory = torch.einsum("bkhw,kd->bdhw", assignment.float(), token_value.float())
        return memory.to(dtype=output_dtype)

    @staticmethod
    def _route_alignment_loss(pet_assignment: torch.Tensor, ct_assignment: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        eps = 1e-8
        loss_map = (pet_assignment.detach().float().clamp_min(eps) * (pet_assignment.detach().float().clamp_min(eps).log() - ct_assignment.float().clamp_min(eps).log())).sum(dim=1, keepdim=True)
        balanced, fg_loss, bg_loss, valid_ratio = _balanced_fg_bg_loss(loss_map, mask)
        return balanced, {
            "writer_route_loss_fg": fg_loss,
            "writer_route_loss_bg": bg_loss,
            "writer_fg_valid_ratio": valid_ratio,
        }

    @staticmethod
    def _memory_grounding_loss(pet_memory: torch.Tensor, pet_query: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        memory_norm = F.normalize(pet_memory.float(), dim=1, eps=1e-6)
        pet_target = F.normalize(pet_query.detach().float(), dim=1, eps=1e-6)
        loss_map = 1.0 - (memory_norm * pet_target).sum(dim=1, keepdim=True)
        balanced, fg_loss, bg_loss, _ = _balanced_fg_bg_loss(loss_map, mask)
        return balanced, {
            "writer_mem_loss_fg": fg_loss,
            "writer_mem_loss_bg": bg_loss,
        }

    def forward_full(self, ct_feats: Sequence[torch.Tensor], pet_feats: Sequence[torch.Tensor], mask: torch.Tensor) -> Tuple[None, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self._validate_feature_list(ct_feats, "ct_feats")
        self._validate_feature_list(pet_feats, "pet_feats")
        writer_index = self.writer_stage - 1
        ct_query = self.stage_modules[str(self.writer_stage)](ct_feats[writer_index])
        pet_query = self.pet_query_proj_writer(pet_feats[writer_index])
        token_key, token_value = self._token_key_value(detach=False)
        ct_assignment = self._assignment(ct_query, token_key)
        pet_assignment = self._assignment(pet_query, token_key)
        pet_memory = self._read_memory(pet_assignment, token_value, output_dtype=pet_query.dtype)

        route_loss, route_diag = self._route_alignment_loss(pet_assignment, ct_assignment, mask)
        mem_loss, mem_diag = self._memory_grounding_loss(pet_memory, pet_query, mask)
        aux_losses = {"route_loss": route_loss, "mem_loss": mem_loss}
        token_key_cosine_offdiag = _mean_off_diagonal_token_cosine(token_key).detach()
        token_value_cosine_offdiag = _mean_off_diagonal_token_cosine(token_value).detach()
        diagnostics = {
            "writer_stage": torch.tensor(float(self.writer_stage), device=ct_query.device),
            "ct_route_entropy": _normalized_assignment_entropy(ct_assignment).detach(),
            "ct_route_peak": _assignment_peak(ct_assignment).detach(),
            "pet_route_entropy": _normalized_assignment_entropy(pet_assignment).detach(),
            "pet_route_peak": _assignment_peak(pet_assignment).detach(),
            "token_key_cosine_offdiag": token_key_cosine_offdiag,
            "token_value_cosine_offdiag": token_value_cosine_offdiag,
            "token_cosine_offdiag": token_value_cosine_offdiag,
            "pet_memory_rms": _rms(pet_memory).detach(),
            "route_loss": route_loss.detach(),
            "mem_loss": mem_loss.detach(),
        }
        diagnostics.update(route_diag)
        diagnostics.update(mem_diag)
        return aux_losses, diagnostics

    def forward_missing(self, ct_feats: Sequence[torch.Tensor], detach_bank: bool = True) -> Tuple[Dict[int, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self._validate_feature_list(ct_feats, "ct_feats")
        retrieved_memories: Dict[int, torch.Tensor] = {}
        diagnostics: Dict[str, torch.Tensor] = {}
        token_key, token_value = self._token_key_value(detach=detach_bank)
        for stage_number, stage_index in zip(self.active_stage_numbers, self.active_stage_indices):
            ct_query = self.stage_modules[str(stage_number)](ct_feats[stage_index])
            ct_assignment = self._assignment(ct_query, token_key)
            retrieved_memory = self._read_memory(ct_assignment, token_value, output_dtype=ct_feats[stage_index].dtype)
            retrieved_memories[stage_number] = retrieved_memory
            stage_diagnostics = {
                "ct_route_entropy": _normalized_assignment_entropy(ct_assignment).detach(),
                "ct_route_peak": _assignment_peak(ct_assignment).detach(),
                "retrieved_memory_rms": _rms(retrieved_memory).detach(),
            }
            diagnostics.update(self._prefix_diagnostics(stage_number, stage_diagnostics))
        diagnostics.update({
            "pg_mtr_writer_stage": torch.tensor(float(self.writer_stage), device=ct_feats[0].device),
            "pg_mtr_shared_token_key_cosine": _mean_off_diagonal_token_cosine(token_key).detach(),
            "pg_mtr_shared_token_value_cosine": _mean_off_diagonal_token_cosine(token_value).detach(),
        })
        return retrieved_memories, {}, diagnostics

    def forward(self, ct_feats: Sequence[torch.Tensor], pet_feats: Optional[Sequence[torch.Tensor]] = None, mode: str = "missing", mask: Optional[torch.Tensor] = None, detach_bank: bool = True):
        mode = str(mode).lower()
        if mode == "full":
            if pet_feats is None or mask is None:
                raise ValueError("mode='full' requires pet_feats and mask")
            return self.forward_full(ct_feats, pet_feats, mask)
        if mode == "missing":
            return self.forward_missing(ct_feats, detach_bank=detach_bank)
        raise ValueError(f"Unsupported PG-MTR mode={mode!r}")
