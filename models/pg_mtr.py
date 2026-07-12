# -*- coding: utf-8 -*-
"""PG-MTR retrieval-only modules for dual-decoder PET-grounded segmentation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

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

        self.memory_tokens = nn.Parameter(torch.empty(self.num_tokens, self.latent_dim))
        self.token_key = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.token_value = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.ct_query_proj = nn.Sequential(
            nn.GroupNorm(_valid_group_count(self.in_channels), self.in_channels),
            nn.Conv2d(self.in_channels, self.latent_dim, kernel_size=1, bias=False),
        )
        self.pet_query_proj = nn.Sequential(
            nn.GroupNorm(_valid_group_count(self.in_channels), self.in_channels),
            nn.Conv2d(self.in_channels, self.latent_dim, kernel_size=1, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _trunc_normal_(self.memory_tokens, std=0.02)
        with torch.no_grad():
            self.memory_tokens.sub_(self.memory_tokens.mean(dim=0, keepdim=True))
        nn.init.xavier_uniform_(self.token_key.weight)
        nn.init.xavier_uniform_(self.token_value.weight)
        for module in (self.ct_query_proj, self.pet_query_proj):
            for submodule in module.modules():
                if isinstance(submodule, nn.Conv2d):
                    nn.init.kaiming_normal_(submodule.weight, mode="fan_out", nonlinearity="linear")
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    def _validate_ct(self, ct_feat: torch.Tensor) -> None:
        if ct_feat.ndim != 4:
            raise ValueError(f"ct_feat must have shape [B,C,H,W], got {tuple(ct_feat.shape)}")
        if ct_feat.shape[1] != self.in_channels:
            raise ValueError(f"Expected CT channels={self.in_channels}, got {ct_feat.shape[1]}")

    def _prepare_pet(self, pet_feat: torch.Tensor, ct_feat: torch.Tensor) -> torch.Tensor:
        if pet_feat.ndim != 4:
            raise ValueError(f"pet_feat must have shape [B,C,H,W], got {tuple(pet_feat.shape)}")
        if pet_feat.shape[0] != ct_feat.shape[0]:
            raise ValueError(f"CT/PET batch size mismatch: {ct_feat.shape[0]} vs {pet_feat.shape[0]}")
        if pet_feat.shape[1] != self.in_channels:
            raise ValueError(f"Expected PET channels={self.in_channels}, got {pet_feat.shape[1]}")
        if pet_feat.shape[-2:] != ct_feat.shape[-2:]:
            pet_feat = F.interpolate(pet_feat, size=ct_feat.shape[-2:], mode="bilinear", align_corners=False)
        return pet_feat

    def _token_key_value(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.token_key(self.memory_tokens), self.token_value(self.memory_tokens)

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
    def _route_alignment_loss(pet_assignment: torch.Tensor, ct_assignment: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        pet_target = pet_assignment.detach().float().clamp_min(eps)
        ct_prediction = ct_assignment.float().clamp_min(eps)
        return (pet_target * (pet_target.log() - ct_prediction.log())).sum(dim=1).mean()

    @staticmethod
    def _memory_grounding_loss(pet_memory: torch.Tensor, pet_query: torch.Tensor) -> torch.Tensor:
        memory_norm = F.normalize(pet_memory.float(), dim=1, eps=1e-6)
        pet_target = F.normalize(pet_query.detach().float(), dim=1, eps=1e-6)
        return 1.0 - (memory_norm * pet_target).sum(dim=1).mean()

    def forward_full(self, ct_feat: torch.Tensor, pet_feat: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self._validate_ct(ct_feat)
        pet_feat = self._prepare_pet(pet_feat, ct_feat)
        ct_source = ct_feat.detach()
        pet_source = pet_feat.detach()
        ct_query = self.ct_query_proj(ct_source)
        pet_query = self.pet_query_proj(pet_source)
        token_key, token_value = self._token_key_value()
        ct_assignment = self._assignment(ct_query, token_key)
        pet_assignment = self._assignment(pet_query, token_key)
        pet_memory = self._read_memory(pet_assignment, token_value, output_dtype=pet_query.dtype)
        route_loss = self._route_alignment_loss(pet_assignment, ct_assignment)
        mem_loss = self._memory_grounding_loss(pet_memory, pet_query)
        aux_losses = {"route_loss": route_loss, "mem_loss": mem_loss}
        diagnostics = {
            "ct_route_entropy": _normalized_assignment_entropy(ct_assignment).detach(),
            "ct_route_peak": _assignment_peak(ct_assignment).detach(),
            "pet_route_entropy": _normalized_assignment_entropy(pet_assignment).detach(),
            "pet_route_peak": _assignment_peak(pet_assignment).detach(),
            "token_cosine_offdiag": _mean_off_diagonal_token_cosine(token_value).detach(),
            "pet_memory_rms": _rms(pet_memory).detach(),
            "route_loss": route_loss.detach(),
            "mem_loss": mem_loss.detach(),
        }
        return aux_losses, diagnostics

    def forward_missing(self, ct_feat: torch.Tensor, detach_bank: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate_ct(ct_feat)
        ct_query = self.ct_query_proj(ct_feat)
        token_key, token_value = self._token_key_value()
        if detach_bank:
            token_key = token_key.detach()
            token_value = token_value.detach()
        ct_assignment = self._assignment(ct_query, token_key)
        retrieved_memory = self._read_memory(ct_assignment, token_value, output_dtype=ct_feat.dtype)
        diagnostics = {
            "ct_route_entropy": _normalized_assignment_entropy(ct_assignment).detach(),
            "ct_route_peak": _assignment_peak(ct_assignment).detach(),
            "token_cosine_offdiag": _mean_off_diagonal_token_cosine(token_key).detach(),
            "retrieved_memory_rms": _rms(retrieved_memory).detach(),
        }
        return retrieved_memory, diagnostics

    def forward(self, ct_feat: torch.Tensor, pet_feat: Optional[torch.Tensor] = None, mode: str = "missing", detach_bank: bool = True):
        mode = str(mode).lower()
        if mode == "full":
            if pet_feat is None:
                raise ValueError("mode='full' requires pet_feat")
            return self.forward_full(ct_feat, pet_feat)
        if mode == "missing":
            return self.forward_missing(ct_feat, detach_bank=detach_bank)
        raise ValueError(f"Unsupported PG-MTR mode={mode!r}")


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
        self.stage_modules = nn.ModuleDict()
        for stage_number, stage_index in zip(self.active_stage_numbers, self.active_stage_indices):
            self.stage_modules[str(stage_number)] = StagePETGroundedMetabolicTokenRetrieval(
                in_channels=channels[stage_index],
                num_tokens=self.num_tokens,
                temperature=self.temperature,
            )

    @staticmethod
    def _validate_feature_list(features: Sequence[torch.Tensor], name: str) -> None:
        if len(features) != 4:
            raise ValueError(f"{name} must contain four stage features, got {len(features)}")

    @staticmethod
    def _prefix_diagnostics(stage_number: int, diagnostics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {f"pg_mtr_s{stage_number}_{key}": value for key, value in diagnostics.items()}

    def forward_full(self, aligned_ct_feats: Sequence[torch.Tensor], pet_feats: Sequence[torch.Tensor]) -> Tuple[None, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self._validate_feature_list(aligned_ct_feats, "aligned_ct_feats")
        self._validate_feature_list(pet_feats, "pet_feats")
        route_losses, memory_losses, diagnostics = [], [], {}
        for stage_number, stage_index in zip(self.active_stage_numbers, self.active_stage_indices):
            stage_losses, stage_diagnostics = self.stage_modules[str(stage_number)].forward_full(aligned_ct_feats[stage_index], pet_feats[stage_index])
            route_losses.append(stage_losses["route_loss"])
            memory_losses.append(stage_losses["mem_loss"])
            diagnostics.update(self._prefix_diagnostics(stage_number, stage_diagnostics))
        route_loss = torch.stack(route_losses).mean()
        memory_loss = torch.stack(memory_losses).mean()
        diagnostics.update({"pg_mtr_route_loss": route_loss.detach(), "pg_mtr_mem_loss": memory_loss.detach()})
        return None, {"pg_mtr_route_loss": route_loss, "pg_mtr_mem_loss": memory_loss}, diagnostics

    def forward_missing(self, aligned_ct_feats: Sequence[torch.Tensor], detach_bank: bool = True) -> Tuple[Dict[int, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        self._validate_feature_list(aligned_ct_feats, "aligned_ct_feats")
        retrieved_memories: Dict[int, torch.Tensor] = {}
        diagnostics: Dict[str, torch.Tensor] = {}
        for stage_number, stage_index in zip(self.active_stage_numbers, self.active_stage_indices):
            memory, stage_diagnostics = self.stage_modules[str(stage_number)].forward_missing(aligned_ct_feats[stage_index], detach_bank=detach_bank)
            retrieved_memories[stage_number] = memory
            diagnostics.update(self._prefix_diagnostics(stage_number, stage_diagnostics))
        return retrieved_memories, {}, diagnostics

    def forward(self, aligned_ct_feats: Sequence[torch.Tensor], pet_feats: Optional[Sequence[torch.Tensor]] = None, mode: str = "missing", detach_bank: bool = True):
        mode = str(mode).lower()
        if mode == "full":
            if pet_feats is None:
                raise ValueError("mode='full' requires pet_feats")
            return self.forward_full(aligned_ct_feats, pet_feats)
        if mode == "missing":
            return self.forward_missing(aligned_ct_feats, detach_bank=detach_bank)
        raise ValueError(f"Unsupported PG-MTR mode={mode!r}")
