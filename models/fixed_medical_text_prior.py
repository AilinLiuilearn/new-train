"""Fixed medical text expert prior for Cross-Scale Shared TaskMoE.

Encodes two fixed CT-PET evidence descriptions once with a frozen local
biomedical text tower, then maps the selected embedding to shared-expert
semantic logits (num_experts). Text never modifies features directly;
it only influences Expert routing decisions.
"""

from __future__ import annotations

import gc
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


FULL_TEXT = (
    "This fused feature combines CT structural information with detailed, "
    "patient-specific metabolic information from real PET."
)

MISSING_TEXT = (
    "This fused feature combines CT structural information with smooth and "
    "coarse tumor-related information from compensated PET."
)


def _require_local_dir(path: str, label: str) -> str:
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f'{label} directory not found (local_files_only): {path}'
        )
    return path


def _encode_fixed_texts(
    text_tower_path: str,
    max_length: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Encode Full/Missing texts once; return L2-normalized [1,D] embeddings."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            'transformers is required for FixedMedicalTextExpertPrior'
        ) from exc

    text_tower_path = _require_local_dir(text_tower_path, 'text_tower')
    required = ('config.json', 'pytorch_model.bin', 'vocab.txt')
    missing = [
        name for name in required
        if not os.path.isfile(os.path.join(text_tower_path, name))
    ]
    if missing:
        raise FileNotFoundError(
            'Local biomedical text tower is incomplete '
            f'(local_files_only=True; will NOT download). '
            f'path={text_tower_path} missing={missing}'
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            text_tower_path,
            local_files_only=True,
            use_fast=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            text_tower_path,
            local_files_only=True,
            use_fast=False,
        )

    text_model = AutoModel.from_pretrained(
        text_tower_path,
        local_files_only=True,
    )
    text_model.eval()
    for p in text_model.parameters():
        p.requires_grad = False

    texts = [FULL_TEXT, MISSING_TEXT]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors='pt',
    )
    with torch.no_grad():
        outputs = text_model(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
        )
        hidden = outputs.last_hidden_state  # [2, L, D]
        mask = encoded['attention_mask'].unsqueeze(-1).to(dtype=hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / denom
        pooled = F.normalize(pooled.float(), p=2, dim=-1, eps=1e-6)
        if not torch.isfinite(pooled).all():
            raise RuntimeError('Fixed text embeddings contain NaN/Inf')

    e_full = pooled[0:1].detach().contiguous()
    e_missing = pooled[1:2].detach().contiguous()
    dim = int(e_full.shape[-1])

    del text_model
    del tokenizer
    del outputs
    del hidden
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return e_full, e_missing, dim


class FixedMedicalTextExpertPrior(nn.Module):
    """Frozen fixed-text embeddings + trainable text-to-expert projection."""

    def __init__(
        self,
        num_experts: int = 6,
        text_model_path: Optional[str] = None,
        text_tower_path: Optional[str] = None,
        max_length: int = 64,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError(f'num_experts must be >= 1, got {num_experts}')

        self.num_experts = int(num_experts)
        self.full_text = FULL_TEXT
        self.missing_text = MISSING_TEXT
        self.backend = 'transformers/AutoModel'
        self.local_only = True
        self.text_encoder_trainable = False
        self.text_encoder_retained = False

        default_clip = (
            '/root/autodl-tmp/mkd-main/new-train/pretrained/biomedclip_model'
        )
        default_tower = (
            '/root/autodl-tmp/mkd-main/new-train/pretrained/biomedbert_text_tower'
        )
        self.biomedclip_model_path = _require_local_dir(
            text_model_path or default_clip,
            'biomedclip_model',
        )
        self.text_tower_path = _require_local_dir(
            text_tower_path or default_tower,
            'text_tower',
        )

        e_full, e_missing, dim = _encode_fixed_texts(
            self.text_tower_path,
            max_length=max_length,
        )
        self.text_embedding_dim = dim
        self.register_buffer('full_text_embedding', e_full, persistent=True)
        self.register_buffer('missing_text_embedding', e_missing, persistent=True)

        self.text_to_expert = nn.Linear(dim, self.num_experts, bias=False)
        nn.init.normal_(self.text_to_expert.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def embedding_cosine(self) -> float:
        a = self.full_text_embedding.float().reshape(-1)
        b = self.missing_text_embedding.float().reshape(-1)
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1, eps=1e-6)
        return float(cos.item())

    def select_embeddings(
        self,
        batch_size: int,
        route: str,
        pet_available: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return [B, D_text] embeddings for the given route."""
        if device is None:
            device = self.full_text_embedding.device
        e_full = self.full_text_embedding.to(device=device, dtype=dtype)
        e_missing = self.missing_text_embedding.to(device=device, dtype=dtype)
        route = str(route).strip().lower()

        if route == 'full':
            return e_full.expand(batch_size, -1).contiguous()
        if route == 'missing':
            return e_missing.expand(batch_size, -1).contiguous()
        if route == 'auto':
            if pet_available is None:
                raise ValueError('route=auto requires pet_available')
            availability = pet_available.to(device=device).long().view(-1)
            if availability.numel() != batch_size:
                raise ValueError(
                    f'pet_available must have shape [B]={batch_size}, '
                    f'got {tuple(pet_available.shape)}'
                )
            if not torch.all((availability == 0) | (availability == 1)):
                raise ValueError('pet_available values must be 0 or 1')
            # 1 -> Full text, 0 -> Missing text
            out = torch.where(
                availability.view(-1, 1).bool(),
                e_full.expand(batch_size, -1),
                e_missing.expand(batch_size, -1),
            )
            return out.contiguous()
        raise ValueError(
            f'Unsupported text prior route={route!r}; use full|missing|auto'
        )

    def forward(
        self,
        batch_size: int,
        route: str,
        pet_available: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Return expert prior logits [B, num_experts] and diagnostics."""
        emb = self.select_embeddings(
            batch_size=batch_size,
            route=route,
            pet_available=pet_available,
            device=device,
            dtype=torch.float32,
        )
        if not torch.isfinite(emb).all():
            raise RuntimeError('Selected text embeddings contain NaN/Inf')

        logits = self.text_to_expert(emb).float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        if not torch.isfinite(logits).all():
            raise RuntimeError('text_to_expert logits contain NaN/Inf')

        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        stats = {
            'text_prior_enabled': torch.tensor(1.0, device=logits.device),
            'text_expert_logits_mean': logits.detach().mean(dim=0),
            'text_expert_probs_mean': probs.detach().mean(dim=0),
            'text_expert_entropy': entropy.detach(),
            'text_embedding_dim': torch.tensor(
                float(self.text_embedding_dim), device=logits.device
            ),
            'full_missing_text_cosine': torch.tensor(
                self.embedding_cosine(), device=logits.device
            ),
        }
        return logits, stats
