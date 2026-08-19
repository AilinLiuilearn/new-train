"""Fixed Real/Proxy PET source text encoder backed by local BioMedCLIP."""
from __future__ import annotations

import gc
import json
import os
from typing import Optional

import torch
import torch.nn as nn

REAL_PROMPT = (
    "Real PET provides patient-specific lesion metabolic information for tumor segmentation."
)
PROXY_PROMPT = (
    "Compensated PET provides shared metabolic information and lesion priors for tumor segmentation."
)


class FixedPETSourceTextEncoder(nn.Module):
    """
    Load local BioMedCLIP once, encode fixed Real/Proxy prompts, cache embeddings.

    Frozen mode (default): encode at init, release the CLIP model, keep buffers only.
    Trainable mode: not supported in the current training stack.
    """

    def __init__(
        self,
        model_path: str,
        text_tower_path: Optional[str] = None,
        trainable: bool = False,
    ):
        super().__init__()
        self.model_path = os.path.abspath(str(model_path))
        self.text_tower_path = os.path.abspath(text_tower_path) if text_tower_path else None
        self.trainable = bool(trainable)
        self.register_buffer("real_embedding", torch.empty(0), persistent=True)
        self.register_buffer("proxy_embedding", torch.empty(0), persistent=True)
        self._ready = False

        if self.trainable:
            raise NotImplementedError(
                "drbf_text_encoder_trainable=true is not supported yet. "
                "Use frozen BioMedCLIP (default) for stable Stage-2 training."
            )

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"Local BioMedCLIP path not found: {self.model_path}. "
                "Set --drbf_text_encoder_path to a local BioMedCLIP directory."
            )
        for fname in ("open_clip_config.json", "open_clip_pytorch_model.bin"):
            fpath = os.path.join(self.model_path, fname)
            if not os.path.isfile(fpath):
                raise FileNotFoundError(
                    f"Missing {fname} under BioMedCLIP path: {self.model_path}"
                )

        self._resolve_text_tower_path()
        self.ensure_ready()

    def _resolve_text_tower_path(self) -> None:
        if self.text_tower_path is not None:
            if not os.path.isdir(self.text_tower_path):
                raise FileNotFoundError(
                    f"Local text tower path not found: {self.text_tower_path}"
                )
            return
        sibling = os.path.join(os.path.dirname(self.model_path), "biomedbert_text_tower")
        if os.path.isdir(sibling):
            self.text_tower_path = sibling
            return
        raise FileNotFoundError(
            "Could not resolve local BioMedBERT text tower. "
            f"Expected sibling directory: {sibling}"
        )

    @property
    def text_dim(self) -> int:
        self.ensure_ready()
        return int(self.real_embedding.numel())

    @property
    def real_prompt_ready(self) -> bool:
        return self._ready and self.real_embedding.numel() > 0

    @property
    def proxy_prompt_ready(self) -> bool:
        return self._ready and self.proxy_embedding.numel() > 0

    def ensure_ready(self) -> None:
        if self._ready:
            return
        self._encode_frozen_once()

    @torch.no_grad()
    def _encode_frozen_once(self) -> None:
        try:
            from open_clip.factory import load_checkpoint
            from open_clip.model import CustomTextCLIP
            from open_clip.tokenizer import HFTokenizer
        except ImportError as exc:
            raise ImportError(
                "BioMedCLIP text encoding requires open_clip_torch and transformers."
            ) from exc

        cfg_path = os.path.join(self.model_path, "open_clip_config.json")
        ckpt_path = os.path.join(self.model_path, "open_clip_pytorch_model.bin")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_cfg = cfg["model_cfg"]
        text_cfg = model_cfg["text_cfg"]
        text_cfg["hf_model_name"] = self.text_tower_path
        text_cfg["hf_tokenizer_name"] = self.text_tower_path
        text_cfg["hf_model_pretrained"] = False

        model = CustomTextCLIP(**model_cfg)
        load_checkpoint(model, ckpt_path, strict=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        tokenizer = HFTokenizer(
            self.text_tower_path,
            context_length=int(text_cfg.get("context_length", 256)),
        )
        tokens = tokenizer([REAL_PROMPT, PROXY_PROMPT])
        feats = model.encode_text(tokens).detach().float().cpu()
        if feats.shape[0] != 2:
            raise RuntimeError(f"Expected 2 prompt embeddings, got {tuple(feats.shape)}")

        self.real_embedding = feats[0]
        self.proxy_embedding = feats[1]
        self._ready = True

        del model, tokenizer, tokens, feats
        gc.collect()

    def get(
        self,
        batch_size: int,
        mode: str,
        device: torch.device,
        dtype: torch.dtype,
        pet_available: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.ensure_ready()
        mode = str(mode).lower().strip()
        real = self.real_embedding.to(device=device, dtype=dtype)
        proxy = self.proxy_embedding.to(device=device, dtype=dtype)

        if mode == "full":
            return real.unsqueeze(0).expand(batch_size, -1)
        if mode == "missing":
            return proxy.unsqueeze(0).expand(batch_size, -1)
        if mode == "auto":
            if pet_available is None:
                raise ValueError("pet_available is required when mode='auto'")
            availability = pet_available.to(device=device).long().view(-1)
            if availability.numel() != batch_size:
                raise ValueError(
                    f"pet_available must have B={batch_size} entries, got {availability.numel()}"
                )
            if not torch.all((availability == 0) | (availability == 1)):
                raise ValueError("pet_available values must be 0 or 1")
            mask = availability.bool().unsqueeze(-1)
            real_b = real.unsqueeze(0).expand(batch_size, -1)
            proxy_b = proxy.unsqueeze(0).expand(batch_size, -1)
            return torch.where(mask, real_b, proxy_b)
        raise ValueError("mode must be 'full', 'missing', or 'auto'")
