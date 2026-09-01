"""FGMS Stage2 wrapper: frozen Stage1 + trainable FGMS MoE + copied Stage2 decoder."""

from __future__ import annotations

import copy
import contextlib
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.fbase_guided_modality_specialized_taskmoe import (
    FBaseGuidedCrossScaleModalitySpecializedTaskMoE,
)

STAGE1_CORE_PREFIXES = (
    "enc_ct.",
    "enc_pet.",
    "ct_align.",
    "prototype_memory.",
    "pet_calibration.",
    "fusion.",
    "decoder.",
)

EXPECTED_CHANNELS = (64, 128, 320, 512)


def _disable_amp_context(device: torch.device):
    """FGMS expert bank mixes fp32 accumulators with fp16 expert outputs under AMP."""
    if device.type == 'cuda':
        return torch.cuda.amp.autocast(enabled=False)
    return contextlib.nullcontext()


def _unwrap_state_dict(checkpoint: dict) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _is_core_key(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in STAGE1_CORE_PREFIXES)


def load_stage1_checkpoint(stage1: nn.Module, checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_state_dict(checkpoint)
    result = stage1.load_state_dict(state_dict, strict=False)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)

    core_missing = [k for k in missing_keys if _is_core_key(k)]
    if core_missing:
        raise RuntimeError(
            f"Stage1 checkpoint missing core keys ({len(core_missing)}): {core_missing[:10]}"
        )

    proto = stage1.prototype_memory
    if not proto.bank_ready:
        raise RuntimeError(
            "Stage1 CPPI prototype bank is not ready. "
            "Provide a fully trained Stage1 checkpoint with populated prototype buffers."
        )

    print(f"[STAGE1] checkpoint path={checkpoint_path}")
    print(f"[STAGE1] checkpoint epoch={checkpoint.get('epoch', 'N/A')}")
    print(f"[STAGE1] best_joint={checkpoint.get('best_joint', 'N/A')}")
    print(f"[STAGE1] best_joint_epoch={checkpoint.get('best_joint_epoch', 'N/A')}")
    print(f"[STAGE1] missing_keys={missing_keys}")
    print(f"[STAGE1] unexpected_keys={unexpected_keys}")
    print(f"[STAGE1] CPPI bank_version={int(proto.bank_version.item())}")
    print(f"[STAGE1] CPPI ready_slots={int(proto.prototype_ready.sum().item())}")

    return checkpoint


def freeze_stage1(stage1: nn.Module) -> None:
    for p in stage1.parameters():
        p.requires_grad = False
    stage1.eval()


def copy_stage2_decoder(stage1_decoder: nn.Module) -> nn.Module:
    stage2_decoder = copy.deepcopy(stage1_decoder)
    for old_p, new_p in zip(stage1_decoder.parameters(), stage2_decoder.parameters()):
        if not torch.equal(old_p.data, new_p.data):
            raise RuntimeError("Stage2 decoder copy mismatch: parameter values differ from Stage1 decoder.")
        if old_p.data_ptr() == new_p.data_ptr():
            raise RuntimeError("Stage2 decoder shares storage with Stage1 decoder; deepcopy failed.")
    for p in stage2_decoder.parameters():
        p.requires_grad = True
    return stage2_decoder


def assert_trainable_parameters(model: "FGMSStage2PETCTModel") -> None:
    moe_trainable = sum(p.numel() for p in model.stage2_moe.parameters() if p.requires_grad)
    dec_trainable = sum(p.numel() for p in model.stage2_decoder.parameters() if p.requires_grad)
    if moe_trainable == 0:
        raise RuntimeError("stage2_moe has no trainable parameters.")
    if dec_trainable == 0:
        raise RuntimeError("stage2_decoder has no trainable parameters.")

    bad = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("stage2_moe.") or name.startswith("stage2_decoder."):
            continue
        bad.append(name)
    if bad:
        raise RuntimeError(
            f"Unexpected trainable parameters outside stage2_moe/stage2_decoder: {bad[:10]}"
        )


def get_cppi_fingerprint(stage1: nn.Module) -> Dict[str, torch.Tensor]:
    proto = stage1.prototype_memory
    fingerprint = {
        "bank_version": proto.bank_version.detach().clone(),
        "prototype_ready": proto.prototype_ready.detach().clone(),
    }
    for scale_idx in range(1, 5):
        fingerprint[f"ct_keys_s{scale_idx}"] = getattr(proto, f"ct_keys_s{scale_idx}").detach().clone()
        fingerprint[f"pet_values_s{scale_idx}"] = getattr(proto, f"pet_values_s{scale_idx}").detach().clone()
    return fingerprint


def assert_cppi_unchanged(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> None:
    for key, val_before in before.items():
        val_after = after[key]
        if not torch.equal(val_before, val_after):
            raise RuntimeError(f"CPPI readonly violation: {key} changed during Stage2 training.")


class FGMSStage2PETCTModel(nn.Module):
    """Frozen Stage1 feature extractor + trainable FGMS MoE + copied Stage2 decoder."""

    is_fgms_stage2 = True

    def __init__(self, stage1: nn.Module, config) -> None:
        super().__init__()
        self.config = config
        self.stage1 = stage1

        channels = tuple(stage1.enc_pet.feature_info.channels())
        if tuple(channels) != EXPECTED_CHANNELS:
            raise ValueError(f"Expected channels {EXPECTED_CHANNELS}, got {channels}")

        self.stage2_moe = FBaseGuidedCrossScaleModalitySpecializedTaskMoE(
            channels=channels,
            expert_dim=int(getattr(config, "fgms_expert_dim", 128)),
            num_experts=int(getattr(config, "fgms_num_experts", 6)),
            top_k=int(getattr(config, "fgms_top_k", 2)),
            atom_num=32,
            atom_dim=256,
            mlp_ratio=2.0,
            enable_balance_loss=bool(getattr(config, "fgms_enable_balance_loss", True)),
            balance_loss_weight=float(getattr(config, "fgms_balance_loss_weight", 0.1)),
            residual_mode=str(getattr(config, "fgms_residual_mode", "zero_start")),
        )
        self.stage2_decoder = copy_stage2_decoder(stage1.decoder)
        assert_trainable_parameters(self)
        self._print_startup_summary()

    @property
    def prototype_memory(self):
        return self.stage1.prototype_memory

    def train(self, mode: bool = True):
        super().train(mode)
        self.stage1.eval()
        if mode:
            self.stage2_moe.train(True)
            self.stage2_decoder.train(True)
        else:
            self.stage2_moe.eval()
            self.stage2_decoder.eval()
        return self

    def _print_startup_summary(self) -> None:
        proto = self.stage1.prototype_memory
        layout = self.stage2_moe.expert_layout()
        total = sum(p.numel() for p in self.parameters())
        stage1_frozen = sum(p.numel() for p in self.stage1.parameters())
        moe_trainable = sum(p.numel() for p in self.stage2_moe.parameters() if p.requires_grad)
        dec_trainable = sum(p.numel() for p in self.stage2_decoder.parameters() if p.requires_grad)
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("[STAGE2] mode=FGMS")
        print(f"[STAGE2] Stage1 checkpoint={getattr(self.config, 'stage1_checkpoint', None)}")
        print("[STAGE2] Stage1 frozen=True")
        print("[STAGE2] CPPI readonly=True")
        print(f"[STAGE2] CPPI bank_version={int(proto.bank_version.item())}")
        print(f"[STAGE2] CPPI ready_slots={int(proto.prototype_ready.sum().item())}")
        print(f"[STAGE2] experts={self.stage2_moe.num_experts}")
        print(f"[STAGE2] expert_layout={layout}")
        print(f"[STAGE2] top_k={self.stage2_moe.top_k}")
        print(f"[STAGE2] balance_loss={self.stage2_moe.enable_balance_loss}")
        print(f"[STAGE2] MoE lr={float(getattr(self.config, 'learning_rate', 6e-5))}")
        print(f"[STAGE2] decoder lr={float(getattr(self.config, 'decoder_lr', 2e-5))}")
        print("[STAGE2] old decoder frozen=True")
        print("[STAGE2] new decoder copied=True")
        print(f"[STAGE2] Total params={total:,}")
        print(f"[STAGE2] Stage1 frozen params={stage1_frozen:,}")
        print(f"[STAGE2] Stage2 MoE trainable params={moe_trainable:,}")
        print(f"[STAGE2] Stage2 Decoder trainable params={dec_trainable:,}")
        print(f"[STAGE2] Total trainable params={total_trainable:,}")

    @torch.no_grad()
    def _extract_full_features(
        self, ct: torch.Tensor, pet: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        ct_feats = self.stage1._encode_ct(ct)
        pet_feats_real = self.stage1._encode_pet(pet)
        _, ct_reference_feats, _ = self.stage1._retrieve_cppi(
            ct_feats,
            compute_report=False,
            save_diagnostics=False,
            print_info=False,
            return_ct_reference=True,
        )
        ct_reference_feats = [x.detach() for x in ct_reference_feats]
        pet_feats_cal = self.stage1.pet_calibration(
            ct_feats,
            pet_feats_real,
            ct_reference_feats,
            reference_valid=True,
        )
        fbase_feats = self.stage1.fusion(ct_feats, pet_feats_cal, mode="full")
        return ct_feats, pet_feats_cal, fbase_feats

    @torch.no_grad()
    def _extract_missing_features(
        self, ct: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        ct_feats = self.stage1._encode_ct(ct)
        pet_feats_proxy, ct_reference_feats, _ = self.stage1._retrieve_cppi(
            ct_feats,
            compute_report=False,
            save_diagnostics=False,
            print_info=False,
            return_ct_reference=True,
        )
        ct_reference_feats = [x.detach() for x in ct_reference_feats]
        pet_feats_cal = self.stage1.pet_calibration(
            ct_feats,
            pet_feats_proxy,
            ct_reference_feats,
            reference_valid=True,
        )
        fbase_feats = self.stage1.fusion(ct_feats, pet_feats_cal, mode="missing")
        return ct_feats, pet_feats_cal, fbase_feats

    def _decode_stage2(self, refined_feats: Sequence[torch.Tensor], target_size) -> dict:
        out = self.stage2_decoder(refined_feats, target_size)
        out["pred"] = out["logits"]
        out["aux"] = {}
        return out

    def _forward_route(self, ct, pet, route: str, target_size):
        if route == "full":
            if pet is None:
                raise ValueError("Full route requires PET input.")
            ct_feats, pet_feats_cal, fbase_feats = self._extract_full_features(ct, pet)
        elif route == "missing":
            ct_feats, pet_feats_cal, fbase_feats = self._extract_missing_features(ct)
        else:
            raise ValueError(f"Unsupported route={route!r}")

        device = ct_feats[0].device
        with _disable_amp_context(device):
            moe_out = self.stage2_moe(
                ct_features=[x.float() for x in ct_feats],
                pet_features=[x.float() for x in pet_feats_cal],
                fbase_features=[x.float() for x in fbase_feats],
                route=route,
            )
            decoder_out = self._decode_stage2(moe_out.features, target_size)
        decoder_out["balance_loss"] = moe_out.balance_loss
        decoder_out["moe_stats"] = moe_out.stats
        return decoder_out

    def _forward_auto(self, ct, pet, pet_available, target_size):
        if pet_available is None:
            pet_available = torch.ones(ct.shape[0], device=ct.device, dtype=torch.long)
        availability = pet_available.to(device=ct.device).long().view(-1)
        if availability.numel() != ct.shape[0]:
            raise ValueError("pet_available must contain one state per sample")
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError("pet_available values must be 0 or 1")

        if torch.all(availability == 1):
            return self._forward_route(ct, pet, "full", target_size)
        if torch.all(availability == 0):
            return self._forward_route(ct, pet, "missing", target_size)

        full_idx = torch.nonzero(availability == 1, as_tuple=False).flatten()
        missing_idx = torch.nonzero(availability == 0, as_tuple=False).flatten()
        batch_size = ct.shape[0]
        logits = ct.new_zeros((batch_size, 1, *target_size))
        balance_loss = ct.new_zeros((), dtype=torch.float32)
        moe_stats = None
        total_weight = 0.0

        if full_idx.numel() > 0:
            out_full = self._forward_route(ct.index_select(0, full_idx), pet.index_select(0, full_idx), "full", target_size)
            logits.index_copy_(0, full_idx, out_full["logits"])
            w = float(full_idx.numel())
            balance_loss = balance_loss + out_full["balance_loss"] * w
            total_weight += w
            moe_stats = out_full["moe_stats"]

        if missing_idx.numel() > 0:
            out_missing = self._forward_route(ct.index_select(0, missing_idx), None, "missing", target_size)
            logits.index_copy_(0, missing_idx, out_missing["logits"])
            w = float(missing_idx.numel())
            balance_loss = balance_loss + out_missing["balance_loss"] * w
            total_weight += w
            moe_stats = out_missing["moe_stats"]

        if total_weight > 0:
            balance_loss = balance_loss / total_weight

        return {
            "logits": logits,
            "pred": logits,
            "aux": {},
            "balance_loss": balance_loss,
            "moe_stats": moe_stats,
        }

    def forward(
        self,
        ct,
        pet=None,
        pet_available=None,
        target_size=None,
        forward_mode: str = "auto",
        mask=None,
    ):
        del mask  # Stage2 never collects CPPI.
        if target_size is None:
            target_size = ct.shape[-2:]
        if forward_mode == "full":
            return self._forward_route(ct, pet, "full", target_size)
        if forward_mode == "missing":
            return self._forward_route(ct, pet, "missing", target_size)
        if forward_mode == "auto":
            return self._forward_auto(ct, pet, pet_available, target_size)
        raise ValueError(f"Unsupported forward_mode={forward_mode!r}")

    def get_cppi_fingerprint(self) -> Dict[str, torch.Tensor]:
        return get_cppi_fingerprint(self.stage1)

    def count_stage1_nonzero_grads(self) -> int:
        count = 0
        for p in self.stage1.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                count += 1
        return count
